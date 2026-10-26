from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SENTIMENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SENTIMENT_DIR.parent
DATA_DIR = SENTIMENT_DIR / "data"


# Combined ground truth (152 old + 447 new, 0 duplicate source_row_id) built in
# the Lexicon_based/Traditional_ML work - same schema as the old 152-row file,
# so load_ground_truth() below needs no changes, just this path swap.
GROUND_TRUTH_PATH = PROJECT_ROOT / "data_news" / "GT.csv"

# The ground truth is joined to this VNCoreNLP word-segmentation of the corpus
# by ``source_row_id`` instead of being re-tokenized here. VNCoreNLP is the
# segmentation PhoBERT was pretrained on, the one E1 (pretrain/) adapts on, and
# the one Lexicon_based / Traditional_ML compare against - so all branches now
# feed PhoBERT identically segmented text.
VNCORENLP_TOKENIZED_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)
SOURCE_ROW_ID_COLUMN = "source_row_id"
TOKENIZED_COLUMN = "Tokenize_content"
# List-of-lists: outer = sentences in the article, inner = tokens in the
# sentence - same column pretrain/domain_adaptive_pretrain.py packs into MLM
# blocks with, reused here to chunk one article at a time for classification
# (see chunk_sentences_into_windows below) instead of silently truncating it.
SENTENCES_COLUMN = "Tokenize_content_sentences"

# Same Tune(730)/Holdout(314) partition of GT.csv that
# Traditional_ML's mentor-plan mục A work uses (built by its
# improve/build_tune_holdout_split.py), joined by source_row_id. Reusing it
# here means PhoBERT-based methods and the sklearn baselines are ever
# evaluated on the identical row sets - a prerequisite for a fair
# cross-method comparison (paired bootstrap / McNemar) on the Holdout rows.
TRADITIONAL_ML_TUNE_HOLDOUT_DIR = (
    PROJECT_ROOT
    / "sentiment_label_method"
    / "Traditional_ML"
    / "Common"
    / "tune_holdout"
)
TUNE_SPLIT_PATH = TRADITIONAL_ML_TUNE_HOLDOUT_DIR / "ground_truth_tune.csv"
HOLDOUT_SPLIT_PATH = TRADITIONAL_ML_TUNE_HOLDOUT_DIR / "ground_truth_holdout.csv"

# E1 output: vinai/phobert-base-v2 after domain-adaptive MLM pretraining on the
# 126k unlabeled equity-news corpus (val perplexity 8.25 -> 3.82). E2 (frozen
# feature probe) and E3 (fine-tune) both start from this instead of the raw base
# or the wonrax e-commerce-review sentiment checkpoint.
BASE_MODEL_PATH = SENTIMENT_DIR / "Model_Output" / "phobert_domain_adapted"

TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"

# Same ordering as Traditional_ML/model/common.py so label ids line up
# across methods when results are compared side by side later.
VALID_LABELS = ["negative", "neutral", "positive"]
LABEL_ALIASES = {
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}

RANDOM_SEED = 42
MAX_LENGTH = 256
VAL_SIZE = 0.2


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    return LABEL_ALIASES.get(str(value).strip().casefold())


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> pd.DataFrame:
    """Load the manually labeled ground truth and attach the VNCoreNLP word
    segmentation of each article (underscore-joined compounds, e.g. "cong_ty"
    not "cong ty" - the format PhoBERT's BPE expects).

    Instead of re-tokenizing ``content`` here, the rows are looked up in
    ``equity_news_tokenized_vncorenlp.parquet`` by ``source_row_id`` (the same
    positional join ``Traditional_ML/prepare_ground_truth.py`` uses). Keeps the
    segmentation identical to E1 pretraining and to the other method branches.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    if not VNCORENLP_TOKENIZED_PATH.exists():
        raise FileNotFoundError(
            f"VNCoreNLP tokenized corpus not found: {VNCORENLP_TOKENIZED_PATH}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {SOURCE_ROW_ID_COLUMN, LABEL_COLUMN, "title"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Ground truth file is missing columns: {sorted(missing_columns)}")
    if df[SOURCE_ROW_ID_COLUMN].isna().any():
        raise ValueError("Ground truth file has null source_row_id values.")

    out = df.copy()
    out[SOURCE_ROW_ID_COLUMN] = out[SOURCE_ROW_ID_COLUMN].astype(int)
    out["ground_truth_label"] = out[LABEL_COLUMN].apply(normalize_label)
    out = out.loc[out["ground_truth_label"].isin(VALID_LABELS)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid labels found in ground truth file.")

    corpus = pd.read_parquet(
        VNCORENLP_TOKENIZED_PATH, columns=["title", TOKENIZED_COLUMN, SENTENCES_COLUMN]
    )
    row_ids = out[SOURCE_ROW_ID_COLUMN].to_numpy()
    if ((row_ids < 0) | (row_ids >= len(corpus))).any():
        raise ValueError(
            f"source_row_id values fall outside the VNCoreNLP corpus (0..{len(corpus) - 1})."
        )
    corpus_rows = corpus.iloc[row_ids].reset_index(drop=True)

    # The join is positional; make sure we pulled the article the annotator saw.
    title_mismatch = (
        out["title"].astype(str).to_numpy() != corpus_rows["title"].astype(str).to_numpy()
    )
    if title_mismatch.any():
        raise ValueError(
            "source_row_id no longer lines up with the VNCoreNLP corpus for "
            f"{int(title_mismatch.sum())} row(s)."
        )

    out[TEXT_COLUMN] = [
        " ".join(str(token) for token in tokens if str(token).strip())
        for tokens in corpus_rows[TOKENIZED_COLUMN]
    ]
    # Kept alongside the flat TEXT_COLUMN so training/scoring code can chunk a
    # long article into <=MAX_LENGTH-token windows (chunk_sentences_into_windows)
    # instead of truncating it - see pretrain/domain_adaptive_pretrain.py for
    # the same column used to pack MLM blocks.
    out[SENTENCES_COLUMN] = list(corpus_rows[SENTENCES_COLUMN])
    out = out.loc[out[TEXT_COLUMN].str.len().gt(0)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No non-empty article text left after joining VNCoreNLP tokens.")

    return out


def load_ground_truth_tune_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split load_ground_truth()'s rows into Traditional_ML's exact Tune/Holdout
    partition (by source_row_id). Raises if the split files are missing or no
    longer add up to the full ground truth (e.g. GT.csv grew
    since the split was built and needs regenerating on the Traditional_ML side
    first).
    """
    if not TUNE_SPLIT_PATH.exists() or not HOLDOUT_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Traditional_ML tune/holdout split not found under "
            f"{TRADITIONAL_ML_TUNE_HOLDOUT_DIR}"
        )

    full = load_ground_truth()
    tune_ids = set(pd.read_csv(TUNE_SPLIT_PATH, encoding="utf-8-sig")[SOURCE_ROW_ID_COLUMN])
    holdout_ids = set(
        pd.read_csv(HOLDOUT_SPLIT_PATH, encoding="utf-8-sig")[SOURCE_ROW_ID_COLUMN]
    )
    overlap = tune_ids & holdout_ids
    if overlap:
        raise ValueError(f"Tune/Holdout source_row_id overlap: {sorted(overlap)[:5]}")

    tune_df = full.loc[full[SOURCE_ROW_ID_COLUMN].isin(tune_ids)].reset_index(drop=True)
    holdout_df = full.loc[full[SOURCE_ROW_ID_COLUMN].isin(holdout_ids)].reset_index(drop=True)
    unmatched = len(full) - len(tune_df) - len(holdout_df)
    if unmatched:
        raise ValueError(
            f"{unmatched} ground-truth row(s) fall outside both Tune and Holdout - "
            "the split is stale relative to GT.csv; regenerate "
            "it on the Traditional_ML side first."
        )
    return tune_df, holdout_df


def build_sentence_strings(sentences) -> list[str]:
    """One article's ``Tokenize_content_sentences`` value -> list of
    space-joined sentence strings (VNCoreNLP tokens), dropping empty
    sentences. Same cleanup pretrain/domain_adaptive_pretrain.py's
    ``load_corpus_sentences`` does per article."""
    if sentences is None:
        return []
    cleaned = []
    for sentence in sentences:
        tokens = [str(token) for token in sentence if str(token).strip()]
        if tokens:
            cleaned.append(" ".join(tokens))
    return cleaned


def pack_sentences_by_length(
    sentence_strings: list[str], sentence_lengths: list[int], max_length: int = MAX_LENGTH
) -> list[str]:
    """Greedily pack one article's sentences into <= max_length BPE-token
    windows without splitting a sentence, given PRE-COMPUTED per-sentence BPE
    lengths - the same packing rule pretrain/domain_adaptive_pretrain.py uses
    for MLM blocks, applied to a single article's sentences instead of the
    whole corpus. Takes lengths as an argument (rather than tokenizing
    inside) so a caller processing many articles can BPE-tokenize every
    sentence in the corpus once, in large batches, instead of once per
    article - see chunk_sentences_into_windows for the single-article
    convenience wrapper that does that tokenization call itself.
    """
    if not sentence_strings:
        return []

    content_len = max_length - 2  # room for the model's bos/eos special tokens
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        if buffer:
            chunks.append(" ".join(buffer))

    for sentence_text, sentence_len in zip(sentence_strings, sentence_lengths):
        if sentence_len > content_len:
            # Rare (~1 in 80k, per the pretrain script's own count): a single
            # sentence alone exceeds a window. Hard-split by whitespace token
            # - an under-estimate of the true BPE-token count since PhoBERT's
            # BPE only ever splits a word into MORE pieces, so every piece
            # still fits; truncation=True at the call site is still the
            # final safety net regardless.
            flush()
            buffer, buffer_len = [], 0
            words = sentence_text.split(" ")
            for start in range(0, len(words), content_len):
                chunks.append(" ".join(words[start : start + content_len]))
            continue

        if buffer_len + sentence_len > content_len:
            flush()
            buffer, buffer_len = [], 0
        buffer.append(sentence_text)
        buffer_len += sentence_len
    flush()
    return chunks


def chunk_sentences_into_windows(
    sentence_strings: list[str], tokenizer, max_length: int = MAX_LENGTH
) -> list[str]:
    """~1/3 of both the 126k corpus and the ground truth rows exceed
    MAX_LENGTH BPE tokens (median ~150-165, but mean ~260-295 and max in the
    thousands), so scoring or training on only
    ``tokenizer(text, truncation=True, max_length=...)`` silently drops
    everything past the first window for those articles. This BPE-tokenizes
    one article's sentences and packs them into <= max_length windows (see
    pack_sentences_by_length) so every window can be scored, then the caller
    pools the per-chunk results back into one article-level result. For many
    articles at once, prefer calling the tokenizer in bulk yourself and using
    pack_sentences_by_length directly - see
    inference/score_corpus_phobert.py::build_article_chunks.
    """
    if not sentence_strings:
        return []
    sentence_lengths = [
        len(ids) for ids in tokenizer(sentence_strings, add_special_tokens=False)["input_ids"]
    ]
    return pack_sentences_by_length(sentence_strings, sentence_lengths, max_length)


def pool_chunks_by_weight(
    chunk_values: np.ndarray, chunk_doc_index: np.ndarray, chunk_weights: np.ndarray, n_docs: int
) -> np.ndarray:
    """Weighted mean of per-chunk vectors (probabilities or embeddings - any
    fixed-width vector) back into one vector per document, weight = chunk
    length in words (a longer chunk represents a bigger share of the
    article). A document with zero chunks gets an all-NaN row - the caller
    decides the fallback (e.g. a uniform label distribution for
    probabilities, since there is no principled default for an embedding).
    """
    n_dims = chunk_values.shape[1]
    weighted_sum = np.zeros((n_docs, n_dims), dtype=np.float64)
    weight_sum = np.zeros(n_docs, dtype=np.float64)
    np.add.at(weighted_sum, chunk_doc_index, chunk_values * chunk_weights[:, None])
    np.add.at(weight_sum, chunk_doc_index, chunk_weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        return weighted_sum / weight_sum[:, None]


def encode_labels(labels: pd.Series) -> list[int]:
    label_to_id = {label: index for index, label in enumerate(VALID_LABELS)}
    return labels.map(label_to_id).tolist()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[true_label, pred_label] += 1
    return matrix


def compute_metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Same precision/recall/f1/accuracy layout as
    Traditional_ML/model/common.py's compute_metrics, kept as a separate
    copy here (rather than imported) so Transfer_Learning stays a
    self-contained method branch, consistent with Lexicon_based and
    Traditional_ML each owning their own evaluation code.
    """
    rows = []
    f1_values = []
    for label_id, label in enumerate(VALID_LABELS):
        true_positive = int(((y_true == label_id) & (y_pred == label_id)).sum())
        false_positive = int(((y_true != label_id) & (y_pred == label_id)).sum())
        false_negative = int(((y_true == label_id) & (y_pred != label_id)).sum())
        support = int((y_true == label_id).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1_score = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1_values.append(f1_score)
        rows.append(
            {
                "metric_scope": label,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": support,
            }
        )

    rows.append(
        {
            "metric_scope": "overall",
            "precision": np.nan,
            "recall": np.nan,
            "f1": float(np.mean(f1_values)),
            "support": int(len(y_true)),
            "accuracy": float((y_true == y_pred).mean()),
        }
    )
    return pd.DataFrame(rows)
