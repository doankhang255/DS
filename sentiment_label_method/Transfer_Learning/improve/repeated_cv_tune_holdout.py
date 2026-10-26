"""Repeated CV on Tune + one-shot Holdout check, mentor plan mục A discipline,
for the Transfer_Learning branch (``--method frozen`` = E2, ``--method
finetune`` = E3).

Why this exists
----------------
``model/frozen_feature_probe.py`` and ``model/finetune_phobert.py`` each run a
single 5-fold CV over the *entire* ground truth and report that as the
headline number. That is exactly the pattern mentor feedback point 1-2 warned
about for the Lexicon_based branch: one split, one number, no sense of how
much it would move on a different split. Traditional_ML's
``improve/repeated_cv.py`` fixed this for the sklearn models (10 repeats of
5-fold CV, mean +/- std) and also built a fixed Tune(730)/Holdout(314) split
(``Traditional_ML/improve/tune_holdout/``) so a final, once-only check exists
that was never touched while anything was being tuned.

This script gives Transfer_Learning the same two things, reusing the exact
same Tune/Holdout row split (by ``source_row_id``, via
``model.common.load_ground_truth_tune_holdout``) so PhoBERT-based methods and
the sklearn baselines are evaluated on identical rows - a prerequisite for
comparing them with a paired bootstrap/McNemar test on the Holdout rows later
(this script only produces the predictions; the paired test itself is a
separate step once Traditional_ML has its own Holdout predictions too).

Self-contained like ``Traditional_ML/improve/repeated_cv.py``: it only
*imports* pure helpers from ``model/common.py``, ``model/frozen_feature_probe.py``
and ``model/finetune_phobert.py`` (tokenization, model construction, the
WeightedLossTrainer, the metric functions) and does its own CV/Holdout wiring
here. It does not edit or monkeypatch any of those files.

Run (from the repo root, using the GPU venv)::

    .../Model_Output/.venv-gpu/Scripts/python.exe \
        News/Build_sentiment_label/Transfer_Learning/improve/repeated_cv_tune_holdout.py \
        --method frozen

    ... --method finetune                      # ~25-30 min (50 fine-tunes + 1 final fit)
    ... --method finetune --repeats 2 --epochs 1 --batch-size 8   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Transfer_Learning.model.common import (  # noqa: E402
    BASE_MODEL_PATH,
    SENTENCES_COLUMN,
    VALID_LABELS,
    compute_metrics_table,
    confusion_matrix,
    encode_labels,
    load_ground_truth_tune_holdout,
)
from News.Build_sentiment_label.Transfer_Learning.model.frozen_feature_probe import (  # noqa: E402
    extract_frozen_features,
)
from News.Build_sentiment_label.Transfer_Learning.model.finetune_phobert import (  # noqa: E402
    WeightedLossTrainer,
    build_chunk_frame,
    build_classifier,
    build_hf_compute_metrics,
    compute_balanced_class_weights,
    make_dataset,
    make_training_args,
    predict_pooled_probabilities,
)


IMPROVE_DIR = Path(__file__).resolve().parent
DATA_DIR = IMPROVE_DIR / "data"

N_SPLITS = 5
N_REPEATS = 10  # matches Traditional_ML/improve/repeated_cv.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", choices=["frozen", "finetune"], required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--C", type=float, default=1.0, help="frozen only: LogisticRegression C")
    parser.add_argument("--epochs", type=float, default=5.0, help="finetune only")
    parser.add_argument("--batch-size", type=int, default=16, help="finetune only")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="finetune only")
    parser.add_argument("--no-amp", action="store_true", help="finetune only: fp32")
    parser.add_argument("--fp16", action="store_true", help="finetune only: force fp16")
    return parser.parse_args()


def per_repeat_row(repeat: int, y: np.ndarray, pred: np.ndarray) -> dict:
    metrics_df = compute_metrics_table(y, pred)
    overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
    per_class = metrics_df.set_index("metric_scope")["f1"]
    return {
        "repeat": repeat,
        "macro_f1": float(overall["f1"]),
        "accuracy": float(overall["accuracy"]),
        "f1_negative": float(per_class["negative"]),
        "f1_neutral": float(per_class["neutral"]),
        "f1_positive": float(per_class["positive"]),
    }


def summarize_repeats(per_repeat_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Same column layout as Traditional_ML/improve/repeated_cv_summary.csv."""
    row = {"model": model_name}
    for column in ("macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"):
        row[f"{column}_mean"] = float(per_repeat_df[column].mean())
        row[f"{column}_std"] = float(per_repeat_df[column].std(ddof=1))
    return pd.DataFrame([row])


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """sklearn's predict_proba columns follow classifier.classes_, which is
    already [0, 1, 2] whenever every class is present in the training fold -
    reordered defensively in case a tiny fold ever drops one.
    """
    aligned = np.zeros((proba.shape[0], len(VALID_LABELS)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        aligned[:, int(class_id)] = proba[:, column]
    return aligned


def build_prediction_frame(
    frame: pd.DataFrame, y: np.ndarray, proba: np.ndarray, pred: np.ndarray
) -> pd.DataFrame:
    out = frame[["id", "source_row_id", "title", "ground_truth_label"]].copy()
    out["predicted_label"] = [VALID_LABELS[index] for index in pred]
    for label_id, label in enumerate(VALID_LABELS):
        out[f"prob_{label}"] = proba[:, label_id]
    out["sentiment_score_ml"] = out["prob_positive"] - out["prob_negative"]
    out["is_correct"] = out["ground_truth_label"].eq(out["predicted_label"])
    return out


def write_holdout_outputs(method: str, prediction_df: pd.DataFrame, y_holdout: np.ndarray) -> None:
    pred = prediction_df["predicted_label"].map({l: i for i, l in enumerate(VALID_LABELS)}).to_numpy()
    metrics_df = compute_metrics_table(y_holdout, pred)
    confusion_df = pd.DataFrame(
        confusion_matrix(y_holdout, pred),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(DATA_DIR / f"{method}_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_df.to_csv(
        DATA_DIR / f"{method}_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    confusion_df.to_csv(
        DATA_DIR / f"{method}_holdout_confusion_matrix.csv", index=False, encoding="utf-8-sig"
    )

    overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
    print(f"\n=== Holdout ({len(y_holdout)} rows, single check) - {method} ===")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print(f"\nHoldout macro F1 : {overall['f1']:.4f}")
    print(f"Holdout accuracy : {overall['accuracy']:.4f}")


# ---------------------------------------------------------------------------
# frozen (E2): repeated CV + holdout on precomputed embeddings
# ---------------------------------------------------------------------------


def run_frozen(args: argparse.Namespace) -> None:
    tune_df, holdout_df = load_ground_truth_tune_holdout()
    y_tune = np.asarray(encode_labels(tune_df["ground_truth_label"]), dtype=int)
    y_holdout = np.asarray(encode_labels(holdout_df["ground_truth_label"]), dtype=int)
    print(f"Tune: {len(tune_df)} rows | Holdout: {len(holdout_df)} rows")

    print("\nExtracting frozen embeddings (chunk + pool, no cache - a few seconds each) ...")
    tune_features = extract_frozen_features(tune_df[SENTENCES_COLUMN])
    holdout_features = extract_frozen_features(holdout_df[SENTENCES_COLUMN])

    print(
        f"\n{args.repeats} repeats x {args.folds}-fold CV on Tune "
        f"(seed = repeat index, LogisticRegression C={args.C}):"
    )
    per_repeat_rows = []
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=repeat)
        oof_pred = np.full(len(y_tune), -1, dtype=int)
        for train_idx, val_idx in splitter.split(tune_features, y_tune):
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced", C=args.C)
            classifier.fit(tune_features[train_idx], y_tune[train_idx])
            proba = align_proba(classifier.predict_proba(tune_features[val_idx]), classifier.classes_)
            oof_pred[val_idx] = proba.argmax(axis=1)
        row = per_repeat_row(repeat, y_tune, oof_pred)
        per_repeat_rows.append(row)
        print(f"  repeat {repeat}: macro-F1 {row['macro_f1']:.4f} | acc {row['accuracy']:.4f}")

    per_repeat_df = pd.DataFrame(per_repeat_rows)
    summary_df = summarize_repeats(per_repeat_df, "phobert_frozen_logreg")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_repeat_df.to_csv(DATA_DIR / "frozen_repeated_cv_per_repeat.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(DATA_DIR / "frozen_repeated_cv_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Repeated CV summary (Tune, {len(y_tune)} rows) ===")
    print(summary_df.to_string(index=False))

    print("\nFitting final LogisticRegression on all of Tune, checking Holdout once ...")
    final_classifier = LogisticRegression(max_iter=1000, class_weight="balanced", C=args.C)
    final_classifier.fit(tune_features, y_tune)
    holdout_proba = align_proba(
        final_classifier.predict_proba(holdout_features), final_classifier.classes_
    )
    holdout_pred = holdout_proba.argmax(axis=1)
    prediction_df = build_prediction_frame(holdout_df, y_holdout, holdout_proba, holdout_pred)
    write_holdout_outputs("frozen", prediction_df, y_holdout)


# ---------------------------------------------------------------------------
# finetune (E3): repeated CV + holdout, fine-tuning PhoBERT per fold
# ---------------------------------------------------------------------------


def run_finetune(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = on_cuda and not args.no_amp and not args.fp16 and torch.cuda.is_bf16_supported()
    use_fp16 = on_cuda and not args.no_amp and not use_bf16
    print("Base model :", BASE_MODEL_PATH)
    print("Device     :", torch.cuda.get_device_name(0) if on_cuda else "cpu")

    tune_df, holdout_df = load_ground_truth_tune_holdout()
    tune_df = tune_df.assign(label=encode_labels(tune_df["ground_truth_label"]))
    holdout_df = holdout_df.assign(label=encode_labels(holdout_df["ground_truth_label"]))
    y_tune = np.asarray(tune_df["label"], dtype=int)
    y_holdout = np.asarray(holdout_df["label"], dtype=int)
    print(f"Tune: {len(tune_df)} rows | Holdout: {len(holdout_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"\n{args.repeats} repeats x {args.folds}-fold CV on Tune (seed = repeat index):")
    per_repeat_rows = []
    overall_start = time.perf_counter()
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=repeat)
        oof_pred = np.full(len(y_tune), -1, dtype=int)
        repeat_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"phobert_rcv_r{repeat}_") as scratch:
            for fold_id, (train_idx, val_idx) in enumerate(
                splitter.split(np.zeros(len(y_tune)), y_tune), start=1
            ):
                train_frame = tune_df.iloc[train_idx].reset_index(drop=True)
                val_frame = tune_df.iloc[val_idx].reset_index(drop=True)
                # Class weights from DOCUMENT-level counts (not chunk counts),
                # same reasoning as model/finetune_phobert.py::main.
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
                oof_pred[val_idx] = probabilities.argmax(axis=1)

                del trainer, model
                if on_cuda:
                    torch.cuda.empty_cache()

        row = per_repeat_row(repeat, y_tune, oof_pred)
        per_repeat_rows.append(row)
        print(
            f"  repeat {repeat}: macro-F1 {row['macro_f1']:.4f} | acc {row['accuracy']:.4f} "
            f"({time.perf_counter() - repeat_start:.1f}s)"
        )

    per_repeat_df = pd.DataFrame(per_repeat_rows)
    summary_df = summarize_repeats(per_repeat_df, "phobert_finetune")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_repeat_df.to_csv(
        DATA_DIR / "finetune_repeated_cv_per_repeat.csv", index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(DATA_DIR / "finetune_repeated_cv_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nRepeated CV wall time: {time.perf_counter() - overall_start:.1f}s")
    print(f"\n=== Repeated CV summary (Tune, {len(y_tune)} rows) ===")
    print(summary_df.to_string(index=False))

    print("\nFine-tuning final model on all of Tune, checking Holdout once ...")
    final_start = time.perf_counter()
    class_weights = compute_balanced_class_weights(tune_df["label"].tolist())
    train_chunk_frame = build_chunk_frame(tune_df, tokenizer)
    holdout_chunk_frame = build_chunk_frame(holdout_df, tokenizer)
    train_dataset = make_dataset(train_chunk_frame, tokenizer)
    holdout_dataset = make_dataset(holdout_chunk_frame, tokenizer)
    model = build_classifier()
    with tempfile.TemporaryDirectory(prefix="phobert_rcv_final_") as scratch:
        trainer = WeightedLossTrainer(
            model=model,
            args=make_training_args(Path(scratch), args, use_bf16, use_fp16, on_cuda),
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_metrics=build_hf_compute_metrics(),
            class_weights=class_weights,
        )
        trainer.train()
        holdout_proba = predict_pooled_probabilities(
            trainer, holdout_chunk_frame, holdout_dataset, len(holdout_df)
        )
    holdout_pred = holdout_proba.argmax(axis=1)
    print(f"Final fit + Holdout predict: {time.perf_counter() - final_start:.1f}s")

    prediction_df = build_prediction_frame(holdout_df, y_holdout, holdout_proba, holdout_pred)
    write_holdout_outputs("finetune", prediction_df, y_holdout)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.method == "frozen":
        run_frozen(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    main()
