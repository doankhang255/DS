"""E3 - Fine-tune the domain-adapted PhoBERT on the manually labeled ground truth.

Starts from ``Model_Output/phobert_domain_adapted`` (the E1 DAPT checkpoint, a
``RobertaForMaskedLM``). ``AutoModelForSequenceClassification.from_pretrained(...,
num_labels=3, ignore_mismatched_sizes=True)`` drops the MLM head and initializes
a fresh 3-class classifier head on top of the adapted encoder.

Evaluation is 5-fold stratified cross-validation with out-of-fold predictions -
the same protocol the Traditional_ML branch uses - so the two method families are
compared on an equal footing rather than on a single 80/20 split. Each fold
trains a fresh classifier with ``WeightedLossTrainer`` (balanced-class-weight
CrossEntropyLoss, weights taken from that fold's training rows) and predicts the
held-out fold; the reported metrics / confusion matrix are computed once on the
pooled out-of-fold predictions covering all 152 rows.

After CV, one final model is trained on all rows (class weights from all rows)
and saved to ``Model_Output/phobert_finetuned_ground_truth/`` as the single
downstream inference artifact - the CV numbers are the honest performance
estimate for it.

GPU perf (copied from the E1 ``domain_adaptive_pretrain.py`` script): TF32
matmul, bf16 autocast by default on CUDA (``--fp16`` forces fp16, ``--no-amp``
disables mixed precision), 0 dataloader workers (Windows spawn/pickle overhead
outweighs the gain here), pinned host memory on CUDA.

Run (from the repo root, using the GPU venv)::

    News/Build_sentiment_label/Transfer_Learning/Model_Output/.venv-gpu/Scripts/python.exe \
        News/Build_sentiment_label/Transfer_Learning/model/finetune_phobert.py

Quick smoke test (2 folds, 1 epoch, small batch)::

    ... finetune_phobert.py --folds 2 --epochs 1 --batch-size 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Transfer_Learning.model.common import (
    BASE_MODEL_PATH,
    DATA_DIR,
    MAX_LENGTH,
    RANDOM_SEED,
    SENTENCES_COLUMN,
    TEXT_COLUMN,
    VALID_LABELS,
    build_sentence_strings,
    chunk_sentences_into_windows,
    compute_metrics_table,
    confusion_matrix,
    encode_labels,
    load_ground_truth,
    pool_chunks_by_weight,
)


# 5-fold stratified CV with out-of-fold predictions, matching the protocol the
# Traditional_ML models use, so PhoBERT fine-tuning and the sklearn baselines are
# evaluated identically. The RTX 5060 fine-tunes each ~120-row fold in well under
# a minute, so running the full model 5x + once more on all rows is only a few
# minutes total - a single 80/20 split (the old model_sentiment_v1/v2 protocol)
# is no longer worth the weaker estimate it gives on 152 rows.
OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "Model_Output" / "phobert_finetuned_ground_truth"
)
OUTPUT_METRICS_PATH = DATA_DIR / "phobert_finetuned_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "phobert_finetuned_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "phobert_finetuned_confusion_matrix.csv"

WEIGHT_DECAY = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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
    return parser.parse_args()


def compute_balanced_class_weights(labels: list[int]) -> torch.Tensor:
    label_counts = pd.Series(labels).value_counts().sort_index()
    missing_labels = set(range(len(VALID_LABELS))).difference(label_counts.index)
    if missing_labels:
        raise ValueError(f"Missing labels in training split: {sorted(missing_labels)}")

    total_samples = len(labels)
    num_classes = len(VALID_LABELS)
    weights = [
        total_samples / (num_classes * label_counts[label_id])
        for label_id in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_hf_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics_df = compute_metrics_table(np.asarray(labels), np.asarray(preds))
        overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
        return {
            "accuracy": float(overall["accuracy"]),
            "f1_macro": float(overall["f1"]),
        }

    return compute_metrics


def build_classifier() -> AutoModelForSequenceClassification:
    """A fresh sequence-classification model from the E1 domain-adapted encoder.

    ``ignore_mismatched_sizes=True`` drops the incoming MLM head and randomly
    initializes the 3-way classifier head - the intended behaviour here.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=len(VALID_LABELS),
        label2id={label: index for index, label in enumerate(VALID_LABELS)},
        id2label=dict(enumerate(VALID_LABELS)),
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )


def build_chunk_frame(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """One row per <=MAX_LENGTH-token chunk of every document in `frame`
    (chunk_sentences_into_windows - never splitting a sentence), carrying the
    PARENT document's label - a chunk is a weak-label augmentation, not an
    independently labeled example, since ~1/3 of these rows are longer than
    MAX_LENGTH and would otherwise be silently truncated to their first
    window. `doc_position` (0-based row order in `frame`) and `chunk_weight`
    (word count) let the caller pool per-chunk predictions back into one
    prediction per document with pool_chunks_by_weight.
    """
    rows = []
    for doc_position, (sentences, label) in enumerate(
        zip(frame[SENTENCES_COLUMN], frame["label"])
    ):
        chunks = chunk_sentences_into_windows(build_sentence_strings(sentences), tokenizer, MAX_LENGTH)
        for chunk in chunks:
            rows.append(
                {
                    TEXT_COLUMN: chunk,
                    "label": label,
                    "doc_position": doc_position,
                    "chunk_weight": max(len(chunk.split(" ")), 1),
                }
            )
    return pd.DataFrame(rows)


def predict_pooled_probabilities(
    trainer: Trainer, chunk_frame: pd.DataFrame, chunk_dataset: Dataset, n_docs: int
) -> np.ndarray:
    """Run the model on every chunk, then weighted-mean-pool chunk-level
    probabilities back to one probability vector per document (matching the
    pooling inference/score_corpus_phobert.py applies to the 126k corpus), so
    CV metrics stay comparable to the pre-chunking protocol - one prediction
    per document, not per chunk."""
    chunk_logits = trainer.predict(chunk_dataset).predictions
    chunk_probabilities = torch.softmax(torch.tensor(chunk_logits), dim=-1).numpy()
    doc_probabilities = pool_chunks_by_weight(
        chunk_probabilities.astype(np.float64),
        chunk_frame["doc_position"].to_numpy(),
        chunk_frame["chunk_weight"].to_numpy(dtype=np.float64),
        n_docs,
    )
    empty_doc_mask = np.isnan(doc_probabilities).any(axis=1)
    if empty_doc_mask.any():
        doc_probabilities[empty_doc_mask] = 1.0 / len(VALID_LABELS)
    return doc_probabilities


def make_dataset(frame: pd.DataFrame, tokenizer) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_pandas(frame[[TEXT_COLUMN, "label"]], preserve_index=False)
    dataset = dataset.map(tokenize_batch, batched=True).remove_columns([TEXT_COLUMN])
    return dataset.with_format("torch")


def make_training_args(
    output_dir: Path,
    args: argparse.Namespace,
    use_bf16: bool,
    use_fp16: bool,
    on_cuda: bool,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        bf16=use_bf16,
        fp16=use_fp16,
        # 0 on purpose: on Windows, DataLoader worker processes (spawn + pickle
        # the Arrow dataset) add far more overhead than they save on 152 rows.
        dataloader_num_workers=0,
        dataloader_pin_memory=on_cuda,
        report_to="none",
        seed=RANDOM_SEED,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        # Blackwell has fast TF32 matmul; free, accuracy-neutral speedup.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_bf16 = (
        on_cuda and not args.no_amp and not args.fp16 and torch.cuda.is_bf16_supported()
    )
    use_fp16 = on_cuda and not args.no_amp and not use_bf16
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"

    ground_truth_df = load_ground_truth()
    labels = encode_labels(ground_truth_df["ground_truth_label"])
    ground_truth_df = ground_truth_df.assign(label=labels).reset_index(drop=True)
    y = np.asarray(ground_truth_df["label"], dtype=int)
    n_rows = len(ground_truth_df)

    print("Base model :", BASE_MODEL_PATH)
    print(
        "Device     :",
        torch.cuda.get_device_name(0) if on_cuda else "cpu",
        "| precision:",
        precision,
    )
    print("Documents  :", n_rows)
    print("Label counts:")
    print(ground_truth_df["ground_truth_label"].value_counts().to_string())

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # 5-fold stratified CV -> out-of-fold probabilities for every row
    # ------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED
    )
    oof_proba = np.zeros((n_rows, len(VALID_LABELS)), dtype=np.float64)
    oof_pred = np.full(n_rows, -1, dtype=int)
    fold_of_row = np.full(n_rows, -1, dtype=int)

    cv_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="phobert_cv_") as scratch:
        for fold_id, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_rows), y), start=1
        ):
            fold_start = time.perf_counter()
            fold_of_row[val_idx] = fold_id

            train_frame = ground_truth_df.iloc[train_idx].reset_index(drop=True)
            val_frame = ground_truth_df.iloc[val_idx].reset_index(drop=True)

            # Class weights from DOCUMENT-level counts (not chunk counts) so a
            # class whose articles happen to run longer doesn't get extra
            # implicit weight just from producing more chunks.
            class_weights = compute_balanced_class_weights(train_frame["label"].tolist())
            train_chunk_frame = build_chunk_frame(train_frame, tokenizer)
            val_chunk_frame = build_chunk_frame(val_frame, tokenizer)
            train_dataset = make_dataset(train_chunk_frame, tokenizer)
            val_dataset = make_dataset(val_chunk_frame, tokenizer)

            model = build_classifier()
            trainer = WeightedLossTrainer(
                model=model,
                args=make_training_args(
                    Path(scratch) / f"fold_{fold_id}", args, use_bf16, use_fp16, on_cuda
                ),
                train_dataset=train_dataset,
                data_collator=data_collator,
                compute_metrics=build_hf_compute_metrics(),
                class_weights=class_weights,
            )
            trainer.train()

            probabilities = predict_pooled_probabilities(
                trainer, val_chunk_frame, val_dataset, len(val_frame)
            )
            oof_proba[val_idx] = probabilities
            oof_pred[val_idx] = probabilities.argmax(axis=1)

            fold_acc = float((oof_pred[val_idx] == y[val_idx]).mean())
            print(
                f"Fold {fold_id}/{args.folds}: train={len(train_idx)} "
                f"val={len(val_idx)} acc={fold_acc:.4f} "
                f"({time.perf_counter() - fold_start:.1f}s)"
            )

            del trainer, model
            if on_cuda:
                torch.cuda.empty_cache()

    if (fold_of_row < 0).any():
        raise AssertionError("Some rows were never assigned to a validation fold.")
    cv_wall = time.perf_counter() - cv_start

    # ------------------------------------------------------------------
    # Metrics / outputs from the pooled out-of-fold predictions
    # ------------------------------------------------------------------
    metrics_df = compute_metrics_table(y, oof_pred)

    prediction_df = ground_truth_df[["id", "title", "ground_truth_label"]].copy()
    prediction_df["predicted_label"] = [VALID_LABELS[index] for index in oof_pred]
    for label_id, label in enumerate(VALID_LABELS):
        prediction_df[f"prob_{label}"] = oof_proba[:, label_id]
    prediction_df["sentiment_score_ml"] = (
        prediction_df["prob_positive"] - prediction_df["prob_negative"]
    )
    prediction_df["is_correct"] = prediction_df["ground_truth_label"].eq(
        prediction_df["predicted_label"]
    )
    prediction_df["fold"] = fold_of_row

    confusion_df = pd.DataFrame(
        confusion_matrix(y, oof_pred),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_CONFUSION_MATRIX_PATH, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Final model: retrain on ALL rows, save as the downstream artifact
    # ------------------------------------------------------------------
    print(f"\nTraining final model on all {n_rows} rows ...")
    final_weights = compute_balanced_class_weights(ground_truth_df["label"].tolist())
    final_chunk_frame = build_chunk_frame(ground_truth_df, tokenizer)
    final_dataset = make_dataset(final_chunk_frame, tokenizer)
    final_model = build_classifier()

    final_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="phobert_final_") as final_scratch:
        final_trainer = WeightedLossTrainer(
            model=final_model,
            args=make_training_args(
                Path(final_scratch), args, use_bf16, use_fp16, on_cuda
            ),
            train_dataset=final_dataset,
            data_collator=data_collator,
            compute_metrics=build_hf_compute_metrics(),
            class_weights=final_weights,
        )
        final_trainer.train()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Drop stale checkpoint dirs from earlier (single-split) runs of this
        # script so the saved artifact is just the model + tokenizer.
        for stale in OUTPUT_DIR.glob("checkpoint-*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        final_trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    final_wall = time.perf_counter() - final_start

    overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
    print(f"\n=== {args.folds}-fold cross-validation (out-of-fold, {n_rows} rows) ===")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix (out-of-fold):")
    print(confusion_df.to_string(index=False))
    print(f"\nOOF macro F1 : {overall['f1']:.4f}")
    print(f"OOF accuracy : {overall['accuracy']:.4f}")
    print(
        f"Wall time    : CV {cv_wall:.1f}s + final fit {final_wall:.1f}s "
        f"= {cv_wall + final_wall:.1f}s"
    )
    print("\nSaved final fine-tuned model to:", OUTPUT_DIR)
    print("Output metrics    :", OUTPUT_METRICS_PATH)
    print("Output predictions:", OUTPUT_PREDICTIONS_PATH)
    print("Output confusion  :", OUTPUT_CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
