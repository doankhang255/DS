"""E2 - Frozen-feature extraction + logistic-regression probe (mentor plan, section E step 2).

WHY this step exists
--------------------
The mentor's ordering for the Transfer_Learning branch is explicit: domain-adaptive
pretraining first (E1 - done, val perplexity 8.25 -> 3.82), then a *frozen-feature*
probe, and only a full fine-tune once there are several hundred labels (E3, which
waits on plan step B growing the ground truth). This file is E2.

With only 152 ground-truth articles, back-propagating through all ~135M PhoBERT
parameters (what ``finetune_phobert.py`` does) is unstable: that run's
positive-class F1 was 0.17 on a single 31-row validation split, and the headline
number moves with whichever rows happen to fall in the split. A frozen probe
removes that variance - every article passes through the E1 domain-adapted
encoder exactly once with the weights frozen, the last hidden state is
mean-pooled into one 768-d sentence embedding, and a plain multinomial
``LogisticRegression`` is fit on top. Only ``768 * 3 + 3 = 2307`` parameters are
learned, so 5-fold stratified cross-validation on 152 rows is well-posed and the
macro-F1 estimate is far more reproducible. It is also the honest "what does the
domain-adapted representation already know" baseline that a later full fine-tune
has to beat.

Run (from the repo root, using the GPU venv)::

    News/Build_sentiment_label/Transfer_Learning/Model_Output/.venv-gpu/Scripts/python.exe \
        News/Build_sentiment_label/Transfer_Learning/model/frozen_feature_probe.py

The 152 x 768 embedding matrix is cached to
``Model_Output/phobert_frozen_features.parquet`` keyed (in the parquet's own
schema metadata) by ground-truth size + BASE_MODEL_PATH + a hash of the text, so
re-runs that only change ``--folds`` / ``--C`` skip the GPU forward pass. Force a
fresh pass with ``--rebuild-cache``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer


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


MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Model_Output"
FEATURE_CACHE_PATH = MODEL_OUTPUT_DIR / "phobert_frozen_features.parquet"
CACHE_META_KEY = b"frozen_probe_meta"

OUTPUT_METRICS_PATH = DATA_DIR / "phobert_frozen_probe_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "phobert_frozen_probe_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "phobert_frozen_probe_confusion_matrix.csv"

# Forward-only pass over 152 short docs -> ~5 batches, a few hundred MB of VRAM.
# Kept deliberately small: another agent may be sharing the single 8GB GPU.
EMBED_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="StratifiedKFold splits for the out-of-fold evaluation (default 5).",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="LogisticRegression inverse-regularisation strength (default 1.0).",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore the cached embeddings and re-run the PhoBERT forward pass.",
    )
    return parser.parse_args()


def text_fingerprint(texts: list[str]) -> str:
    hasher = hashlib.sha1()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


# Bumped when the extraction method itself changes (not just the input docs)
# so a cache built by an older version of extract_frozen_features is never
# silently reused - e.g. the switch from "truncate to MAX_LENGTH" to
# "chunk + weighted-mean-pool" below produces different embeddings for the
# ~1/3 of rows longer than MAX_LENGTH even though n_docs/content are unchanged.
EXTRACTION_METHOD_VERSION = "chunk_weighted_mean_v1"


def build_cache_meta(texts: list[str], hidden_size: int | None = None) -> dict:
    """The key that decides whether a cached embedding matrix is still valid:
    same number of docs, same base checkpoint, same truncation length, same
    exact text, same extraction method. Any change -> recompute.
    """
    meta = {
        "n_docs": len(texts),
        "base_model_path": str(BASE_MODEL_PATH),
        "max_length": int(MAX_LENGTH),
        "content_sha1": text_fingerprint(texts),
        "extraction_method": EXTRACTION_METHOD_VERSION,
    }
    if hidden_size is not None:
        meta["hidden_size"] = int(hidden_size)
    return meta


def load_cached_features(expected_meta: dict, expected_ids: list) -> np.ndarray | None:
    if not FEATURE_CACHE_PATH.exists():
        return None
    try:
        table = pq.read_table(FEATURE_CACHE_PATH)
    except (OSError, pa.ArrowInvalid):
        return None
    raw_meta = (table.schema.metadata or {}).get(CACHE_META_KEY)
    if raw_meta is None:
        return None
    try:
        meta = json.loads(raw_meta.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    for key in ("n_docs", "base_model_path", "max_length", "content_sha1", "extraction_method"):
        if meta.get(key) != expected_meta[key]:
            return None
    frame = table.to_pandas()
    if [str(value) for value in frame["id"].tolist()] != [str(value) for value in expected_ids]:
        return None
    dim_columns = [column for column in frame.columns if column.startswith("dim_")]
    if not dim_columns:
        return None
    return frame[dim_columns].to_numpy(dtype=np.float32)


def save_cached_features(features: np.ndarray, ids: list, meta: dict) -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        features, columns=[f"dim_{index}" for index in range(features.shape[1])]
    )
    frame.insert(0, "id", ids)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[CACHE_META_KEY] = json.dumps(meta).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    pq.write_table(table, FEATURE_CACHE_PATH)


@torch.no_grad()
def extract_frozen_features(sentences_column: pd.Series) -> np.ndarray:
    """One frozen forward pass per <=MAX_LENGTH-token CHUNK of each article
    (chunk_sentences_into_windows - never splitting a sentence, instead of
    truncating ~1/3 of the 1044 rows to their first MAX_LENGTH tokens), mean-
    pool ``last_hidden_state`` over the non-padding tokens of each chunk into
    a 768-d vector, then weighted-mean-pool an article's chunk vectors
    (weight = chunk length in words) into one 768-d embedding per article.

    ``AutoModel`` on the E1 ``RobertaForMaskedLM`` checkpoint returns the bare
    RoBERTa encoder; the "some weights ... lm_head ... were not used" warning is
    expected and harmless.
    """
    on_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if on_cuda else "cpu")
    if on_cuda:
        # Blackwell (RTX 5060): accuracy-neutral matmul speedup, same as pretrain/.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = on_cuda and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, use_fast=False, local_files_only=True
    )
    model = AutoModel.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    model.to(device)
    model.eval()

    print(
        "Device        :",
        torch.cuda.get_device_name(0) if on_cuda else "cpu",
        "| forward autocast:",
        "bf16" if use_bf16 else "off",
    )
    print("Hidden size   :", model.config.hidden_size)

    n_docs = len(sentences_column)
    flat_chunks: list[str] = []
    doc_index: list[int] = []
    chunk_weights: list[int] = []
    n_docs_multi_chunk = 0
    for doc_id, sentences in enumerate(sentences_column):
        chunks = chunk_sentences_into_windows(
            build_sentence_strings(sentences), tokenizer, MAX_LENGTH
        )
        if len(chunks) > 1:
            n_docs_multi_chunk += 1
        for chunk in chunks:
            flat_chunks.append(chunk)
            doc_index.append(doc_id)
            chunk_weights.append(max(len(chunk.split(" ")), 1))
    print(
        f"  {len(flat_chunks)} cua so tu {n_docs} bai "
        f"({n_docs_multi_chunk} bai can >1 cua so, tuc bi cat truoc day)"
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )

    chunk_embedding_batches: list[np.ndarray] = []
    for start in range(0, len(flat_chunks), EMBED_BATCH_SIZE):
        batch_texts = flat_chunks[start : start + EMBED_BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,  # dynamic: pad to the longest chunk in THIS batch only
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with autocast_ctx:
            outputs = model(**encoded)
        # Pool in fp32 so bf16 rounding over up to 256 tokens does not leak into
        # the cached features.
        last_hidden = outputs.last_hidden_state.float()  # (batch, seq, hidden)
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (batch, hidden)
        token_counts = mask.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        mean_pooled = summed / token_counts
        chunk_embedding_batches.append(mean_pooled.cpu().numpy().astype(np.float32))
        print(f"  embedded {min(start + EMBED_BATCH_SIZE, len(flat_chunks))}/{len(flat_chunks)} cua so")

    if on_cuda:
        del model
        torch.cuda.empty_cache()

    chunk_embeddings = np.vstack(chunk_embedding_batches).astype(np.float64)
    doc_embeddings = pool_chunks_by_weight(
        chunk_embeddings,
        np.asarray(doc_index, dtype=np.int64),
        np.asarray(chunk_weights, dtype=np.float64),
        n_docs,
    )
    empty_doc_mask = np.isnan(doc_embeddings).any(axis=1)
    if empty_doc_mask.any():
        print(f"  {int(empty_doc_mask.sum())} bai khong co cua so nao -> gan vector 0")
        doc_embeddings[empty_doc_mask] = 0.0
    return doc_embeddings.astype(np.float32)


def run_cross_validation(
    features: np.ndarray, y: np.ndarray, n_folds: int, C: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit LogisticRegression on each training fold, predict proba on the
    held-out fold, and collect out-of-fold predictions for every row.
    """
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED
    )
    n_classes = len(VALID_LABELS)
    oof_proba = np.zeros((len(y), n_classes), dtype=np.float64)
    oof_pred = np.full(len(y), -1, dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, y), start=1):
        classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=C
        )
        classifier.fit(features[train_idx], y[train_idx])
        proba = classifier.predict_proba(features[test_idx])

        # Re-order columns into VALID_LABELS id order (defensive: with all three
        # classes present in every training fold this is already 0, 1, 2).
        aligned = np.zeros((len(test_idx), n_classes), dtype=np.float64)
        for column, class_id in enumerate(classifier.classes_):
            aligned[:, int(class_id)] = proba[:, column]

        fold_pred = aligned.argmax(axis=1)
        oof_proba[test_idx] = aligned
        oof_pred[test_idx] = fold_pred

        fold_metrics = compute_metrics_table(y[test_idx], fold_pred)
        overall = fold_metrics.loc[fold_metrics["metric_scope"].eq("overall")].iloc[0]
        print(
            f"  fold {fold}: train {len(train_idx):3d} / test {len(test_idx):3d} "
            f"| macro-F1 {overall['f1']:.4f} | acc {overall['accuracy']:.4f}"
        )

    if (oof_pred < 0).any():
        raise RuntimeError("Some rows never landed in a held-out fold.")
    return oof_proba, oof_pred


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2 for cross-validation.")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    started = time.perf_counter()

    ground_truth_df = load_ground_truth()
    y = np.asarray(encode_labels(ground_truth_df["ground_truth_label"]), dtype=np.int64)
    texts = ground_truth_df[TEXT_COLUMN].astype(str).tolist()
    ids = ground_truth_df["id"].tolist()

    print("Base model    :", BASE_MODEL_PATH)
    print("Documents     :", len(ground_truth_df))
    print("Label counts  :")
    print(ground_truth_df["ground_truth_label"].value_counts().to_string())

    expected_meta = build_cache_meta(texts)
    features: np.ndarray | None = None
    if not args.rebuild_cache:
        features = load_cached_features(expected_meta, ids)
        if features is not None:
            print(
                f"\nLoaded cached embeddings {features.shape} from "
                f"{FEATURE_CACHE_PATH.name} (forward pass skipped)."
            )

    if features is None:
        print("\nRunning frozen PhoBERT forward pass ...")
        forward_started = time.perf_counter()
        features = extract_frozen_features(ground_truth_df[SENTENCES_COLUMN])
        print(
            f"Forward pass finished in {time.perf_counter() - forward_started:.1f}s "
            f"-> embeddings {features.shape}"
        )
        save_cached_features(
            features, ids, build_cache_meta(texts, features.shape[1])
        )
        print("Cached embeddings to:", FEATURE_CACHE_PATH)

    print(
        f"\n{args.folds}-fold stratified CV "
        f"(seed {RANDOM_SEED}, LogisticRegression C={args.C}, class_weight=balanced):"
    )
    oof_proba, oof_pred = run_cross_validation(features, y, args.folds, args.C)

    metrics_df = compute_metrics_table(y, oof_pred)
    confusion_df = pd.DataFrame(
        confusion_matrix(y, oof_pred),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_CONFUSION_MATRIX_PATH, index=False, encoding="utf-8-sig")

    overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
    print("\nOut-of-fold metrics (all 152 rows):")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix (out-of-fold):")
    print(confusion_df.to_string(index=False))
    print(f"\nMacro F1 : {overall['f1']:.4f}")
    print(f"Accuracy : {overall['accuracy']:.4f}")
    print(f"Elapsed  : {time.perf_counter() - started:.1f}s")
    print("Outputs  :")
    print("  ", OUTPUT_METRICS_PATH)
    print("  ", OUTPUT_PREDICTIONS_PATH)
    print("  ", OUTPUT_CONFUSION_MATRIX_PATH)
    print("  ", FEATURE_CACHE_PATH)


if __name__ == "__main__":
    main()
