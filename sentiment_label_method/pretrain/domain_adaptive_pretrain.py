"""E1 - Domain-adaptive pretraining (DAPT) for PhoBERT on the equity-news corpus.

What it does
------------
1. Read ``equity_news_tokenized_vncorenlp.parquet`` column
   ``Tokenize_content_sentences`` - the article already split into sentences,
   each a list of word-segmented tokens with ``_``-joined compounds (the
   VNCoreNLP/RDRSegmenter segmentation PhoBERT was pretrained on). Articles
   whose ``source_row_id`` appears in ``data_news/GT.csv``
   are dropped first (see ``GROUND_TRUTH_COMBINED_PATH`` above) so the E2/E3
   evaluation sets stay text-unseen by this step, not just label-unseen.
2. BPE-tokenize each sentence, then greedily pack *whole sentences* into blocks
   of up to ``MAX_SEQ_LENGTH`` tokens - never splitting a sentence, never
   crossing an article boundary. Every training sequence is therefore
   linguistically coherent (unlike a blind 256-token window that cuts
   mid-sentence and can merge two unrelated articles), while still staying long
   enough to give MLM real bidirectional context. Rare sentences longer than a
   block are hard-split as a fallback.
3. Hold out ``VAL_FRACTION`` of the blocks to track eval loss / perplexity - the
   signal that DAPT is actually adapting the model to the domain.
4. Train ``AutoModelForMaskedLM`` starting from the local ``vinai/phobert-base-v2``
   snapshot, with dynamic 15% masking (``DataCollatorForLanguageModeling``).
5. Save the adapted model + tokenizer to ``Model_Output/phobert_domain_adapted/``
   (this becomes the base checkpoint for E2 frozen-feature extraction and E3
   full fine-tuning) and append a row to ``data/phobert_dapt_eval.csv``.

The tokenized+blocked dataset is cached under ``Model_Output/phobert_dapt_cache/``
so re-runs (e.g. trying more epochs) skip the preprocessing.

Run (from the repo root, using the GPU venv)::

    sentiment_label_method/Model_Output/.venv-gpu/Scripts/python.exe \
        sentiment_label_method/pretrain/domain_adaptive_pretrain.py

Quick smoke test on 2000 docs / 1 epoch::

    ... domain_adaptive_pretrain.py --max-docs 2000 --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SENTIMENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PARQUET = (
    PROJECT_ROOT
    / "data_news"
    / "data_tokenized"
    / "equity_news_tokenized_vncorenlp.parquet"
)
# List-of-lists: outer = sentences in the article, inner = tokens in the sentence.
SENTENCES_COLUMN = "Tokenize_content_sentences"

# vinai/phobert-base-v2 is the *base* masked-LM (not the wonrax sentiment head)
# - the right starting point for continued MLM pretraining. Already in the local
# HF hub cache, so no download / network needed.
BASE_MODEL_PATH = (
    SENTIMENT_DIR
    / "Model_Output"
    / "hub"
    / "models--vinai--phobert-base-v2"
    / "snapshots"
    / "e2375d266bdf39c6e8e9a87af16a5da3190b0cc8"
)

# Every ground-truth batch (Lexicon_based/Traditional_ML/Transfer_Learning all
# draw from this combined file - 152 old + 447 new, 599 total) has a
# source_row_id pointing at a positional row in INPUT_PARQUET. Excluded from
# the DAPT corpus by default: E2/E3 later evaluate on these exact articles, so
# training MLM on them first would mean the model "read" the evaluation
# articles' text before being scored on them - a mild contamination the user
# caught. No sentiment labels ever reach E1 either way (MLM never sees them),
# but excluding the text too keeps the eval honest. See --include-ground-truth
# to reproduce the earlier (contaminated) run for comparison.
GROUND_TRUTH_COMBINED_PATH = PROJECT_ROOT / "data_news" / "GT.csv"

OUTPUT_DIR = SENTIMENT_DIR / "Model_Output" / "phobert_domain_adapted"
CACHE_DIR = SENTIMENT_DIR / "Model_Output" / "phobert_dapt_cache"
EVAL_CSV = SENTIMENT_DIR / "data" / "phobert_dapt_eval.csv"

# PhoBERT's positional table is 258 wide -> 256 usable tokens after the two
# special tokens. Same length model_sentiment_v1/v2 and finetune_phobert.py use.
MAX_SEQ_LENGTH = 256
MLM_PROBABILITY = 0.15
VAL_FRACTION = 0.05
MIN_TOKENS_PER_DOC = 5
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap the number of articles (smoke tests). Default: use all.",
    )
    parser.add_argument(
        "--map-num-proc",
        type=int,
        default=1,
        help="Processes for datasets.map tokenization. >1 is faster but on "
        "Windows can be flaky; bump only if it works on your box.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision entirely (train in fp32).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 mixed precision instead of the bf16 default (bf16 is "
        "native on this RTX 5060 / Blackwell and needs no loss scaler).",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore any cached tokenized dataset and rebuild it.",
    )
    parser.add_argument(
        "--include-ground-truth",
        action="store_true",
        help="Do NOT exclude ground-truth source_row_id articles from the DAPT "
        "corpus. Off by default; pass this only to reproduce the earlier "
        "contaminated run for a before/after comparison.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint-* under OUTPUT_DIR instead of "
        "starting over (e.g. after a mid-run CUDA/driver crash). Must be run "
        "with the same --max-docs/--epochs/--batch-size/--grad-accum as the "
        "interrupted run, with the cached blocked dataset still on disk, so "
        "the reconstructed train/eval split matches exactly.",
    )
    return parser.parse_args()


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def load_ground_truth_row_ids(path: Path = GROUND_TRUTH_COMBINED_PATH) -> set[int]:
    """source_row_id of every manually labeled article (all batches combined,
    599 as of writing) - positional row indices into INPUT_PARQUET. Empty set
    if the file isn't there yet, so this never hard-fails a run.
    """
    if not path.exists():
        print(f"(no ground-truth file at {path} - nothing to exclude)")
        return set()
    ids = pd.read_csv(path, encoding="utf-8-sig", usecols=["source_row_id"])["source_row_id"]
    return {int(value) for value in ids.dropna().unique()}


def load_corpus_sentences(
    max_docs: int | None, excluded_row_ids: set[int]
) -> list[list[str]]:
    """Return one entry per article: a list of sentence strings (tokens joined
    by spaces). Articles with fewer than MIN_TOKENS_PER_DOC tokens total, or
    whose positional row index is in ``excluded_row_ids``, are dropped.
    """
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Tokenized corpus not found: {INPUT_PARQUET}")

    frame = pd.read_parquet(INPUT_PARQUET, columns=[SENTENCES_COLUMN])
    print(f"Loaded {len(frame):,} rows from {INPUT_PARQUET.name}")

    articles: list[list[str]] = []
    n_excluded = 0
    for row_id, sentences in enumerate(frame[SENTENCES_COLUMN]):
        if row_id in excluded_row_ids:
            n_excluded += 1
            continue
        if sentences is None or len(sentences) == 0:
            continue
        cleaned = []
        total_tokens = 0
        for sentence in sentences:
            tokens = [str(tok) for tok in sentence if str(tok).strip()]
            if tokens:
                cleaned.append(" ".join(tokens))
                total_tokens += len(tokens)
        if cleaned and total_tokens >= MIN_TOKENS_PER_DOC:
            articles.append(cleaned)

    if excluded_row_ids:
        print(f"Excluded {n_excluded:,} / {len(excluded_row_ids):,} ground-truth article(s) "
              f"found in the corpus")
    print(f"Kept {len(articles):,} articles with >= {MIN_TOKENS_PER_DOC} tokens")

    if max_docs is not None:
        articles = articles[:max_docs]
        print(f"--max-docs: truncated to {len(articles):,} articles")

    return articles


def build_block_dataset(
    articles: list[list[str]],
    tokenizer,
    map_num_proc: int,
    rebuild_cache: bool,
    cache_tag: str = "",
) -> Dataset:
    """BPE-tokenize every sentence, then greedily pack whole sentences into
    <= MAX_SEQ_LENGTH blocks without splitting a sentence or crossing an
    article boundary. Cached to disk keyed by corpus size + sequence length
    (+ ``cache_tag``, so an excluded-ground-truth run never loads the cache
    from a run that included it, or vice versa).
    """
    cache_path = CACHE_DIR / f"blocks_sent_{len(articles)}docs_{MAX_SEQ_LENGTH}tok{cache_tag}"
    if cache_path.exists() and not rebuild_cache:
        print(f"Loading cached blocked dataset from {cache_path}")
        return Dataset.load_from_disk(str(cache_path))

    # Flatten sentences (keeping which article each belongs to) so the slow
    # PhoBERT BPE runs once over one batched dataset instead of per article.
    flat_sentences: list[str] = []
    article_of_sentence: list[int] = []
    for article_idx, sentences in enumerate(articles):
        for sentence in sentences:
            flat_sentences.append(sentence)
            article_of_sentence.append(article_idx)

    sentence_dataset = Dataset.from_dict({"text": flat_sentences})

    def tokenize_fn(batch: dict) -> dict:
        return {"ids": tokenizer(batch["text"], add_special_tokens=False)["input_ids"]}

    sentence_dataset = sentence_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=map_num_proc,
        remove_columns=["text"],
        desc="BPE tokenizing sentences",
    )
    sentence_ids = sentence_dataset["ids"]

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    content_len = MAX_SEQ_LENGTH - 2  # room for the leading bos + trailing eos

    input_ids_blocks: list[list[int]] = []
    attention_masks: list[list[int]] = []
    buffer: list[int] = []
    current_article = -1

    def flush() -> None:
        if buffer:
            sequence = [bos, *buffer, eos]
            input_ids_blocks.append(sequence)
            attention_masks.append([1] * len(sequence))
            buffer.clear()

    for article_idx, ids in zip(article_of_sentence, sentence_ids):
        if article_idx != current_article:
            flush()
            current_article = article_idx

        if len(ids) > content_len:
            # Rare (~1 in 80k): a single sentence longer than a block.
            flush()
            for start in range(0, len(ids), content_len):
                piece = ids[start : start + content_len]
                sequence = [bos, *piece, eos]
                input_ids_blocks.append(sequence)
                attention_masks.append([1] * len(sequence))
            continue

        if len(buffer) + len(ids) > content_len:
            flush()
        buffer.extend(ids)
    flush()

    blocks = Dataset.from_dict(
        {"input_ids": input_ids_blocks, "attention_mask": attention_masks}
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    blocks.save_to_disk(str(cache_path))
    print(f"Cached {len(blocks):,} blocks to {cache_path}")
    return blocks


def append_eval_row(row: dict) -> None:
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EVAL_CSV.exists()
    with EVAL_CSV.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        # Blackwell has fast TF32 matmul; free accuracy-neutral speedup.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_bf16 = on_cuda and not args.no_amp and not args.fp16 and torch.cuda.is_bf16_supported()
    use_fp16 = on_cuda and not args.no_amp and not use_bf16
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print("Base model :", BASE_MODEL_PATH)
    print("Device     :", torch.cuda.get_device_name(0) if on_cuda else "cpu", "| precision:", precision)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

    excluded_row_ids = set() if args.include_ground_truth else load_ground_truth_row_ids()
    if excluded_row_ids:
        print(f"Ground-truth exclusion ON: {len(excluded_row_ids):,} source_row_id "
              f"from {GROUND_TRUTH_COMBINED_PATH.name} will be dropped from the DAPT corpus")
    else:
        print("Ground-truth exclusion OFF (--include-ground-truth) - reproducing the "
              "earlier contaminated run")

    articles = load_corpus_sentences(args.max_docs, excluded_row_ids)
    cache_tag = "_exclgt" if excluded_row_ids else ""
    blocks = build_block_dataset(
        articles, tokenizer, args.map_num_proc, args.rebuild_cache, cache_tag
    )

    split = blocks.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    train_dataset = split["train"].with_format("torch")
    eval_dataset = split["test"].with_format("torch")
    print(f"Blocks: {len(blocks):,} total | train {len(train_dataset):,} | val {len(eval_dataset):,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        # 0 on purpose: on Windows, DataLoader worker processes (spawn + pickle
        # the Arrow dataset) add far more overhead than they save here.
        dataloader_num_workers=0,
        dataloader_pin_memory=on_cuda,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR) if args.resume else None
    if args.resume and resume_checkpoint is None:
        raise FileNotFoundError(f"--resume given but no checkpoint-* found under {OUTPUT_DIR}")

    if resume_checkpoint is None:
        baseline = trainer.evaluate()
        baseline_ppl = math.exp(baseline["eval_loss"]) if baseline["eval_loss"] < 20 else float("inf")
        print(f"\nBaseline (phobert-base-v2, no adaptation): eval_loss={baseline['eval_loss']:.4f} "
              f"| perplexity={baseline_ppl:.2f}")
    else:
        baseline = None
        print(f"\nResuming from {resume_checkpoint} - skipping the baseline eval "
              f"(it already ran at the start of the interrupted run).")

    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    final = trainer.evaluate()
    final_ppl = math.exp(final["eval_loss"]) if final["eval_loss"] < 20 else float("inf")
    print(f"\nAfter DAPT: eval_loss={final['eval_loss']:.4f} | perplexity={final_ppl:.2f}")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Saved domain-adapted model to:", OUTPUT_DIR)

    append_eval_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_docs": len(articles),
            "n_blocks": len(blocks),
            "ground_truth_excluded": len(excluded_row_ids),
            "packing": "sentence",
            "precision": precision,
            "max_seq_length": MAX_SEQ_LENGTH,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "baseline_eval_loss": round(baseline["eval_loss"], 5) if baseline else "resumed_run",
            "baseline_perplexity": round(baseline_ppl, 4) if baseline else "resumed_run",
            "final_eval_loss": round(final["eval_loss"], 5),
            "final_perplexity": round(final_ppl, 4),
        }
    )
    print("Appended eval summary to:", EVAL_CSV)


if __name__ == "__main__":
    main()
