"""Mentor plan muc 1: bao cao mean +/- std macro-F1 cua E3 (fine-tune toan
phan) qua 5 seed khac nhau, thay vi 1 con so don le (~1000 nhan, ~135M tham
so cua PhoBERT -> phuong sai theo seed co the lon).

Seed o day chi anh huong 2 thu: cach khoi tao NGAU NHIEN dau phan loai moi
(classifier.dense/out_proj) va THU TU xao tron du lieu train moi epoch -
KHONG anh huong cach chia fold (StratifiedKFold dung CO DINH 1 random_state
cho ca 5 lan chay, xem FOLD_SPLIT_SEED ben duoi) de tach rieng dung 1 nguon
phuong sai ma mentor de cap (phuong sai do fine-tune, khong tron voi phuong
sai do chia fold khac nhau - cai do improve/repeated_cv_tune_holdout.py da
kiem tra roi, la 1 cau hoi khac).

Chay (dung venv GPU cua repo)::

    .../Model_Output/.venv-gpu/Scripts/python.exe \
        News/Build_sentiment_label/Transfer_Learning/improve/seed_variance_finetune.py
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
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, DataCollatorWithPadding

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Transfer_Learning.model.common import (  # noqa: E402
    BASE_MODEL_PATH,
    VALID_LABELS,
    compute_metrics_table,
    encode_labels,
    load_ground_truth,
)
from News.Build_sentiment_label.Transfer_Learning.model.finetune_phobert import (  # noqa: E402
    WeightedLossTrainer,
    build_chunk_frame,
    build_classifier,
    compute_balanced_class_weights,
    make_dataset,
    make_training_args,
    predict_pooled_probabilities,
)

IMPROVE_DIR = Path(__file__).resolve().parent
DATA_DIR = IMPROVE_DIR / "data"

SEEDS = [42, 7, 123, 2024, 99]  # 5 seed, dung theo de xuat "3-5 seed" cua mentor
FOLD_SPLIT_SEED = 42  # CO DINH cho ca 5 lan chay - chi doi seed khoi tao/train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def run_one_seed(
    seed: int,
    ground_truth_df: pd.DataFrame,
    y: np.ndarray,
    tokenizer,
    data_collator,
    args: argparse.Namespace,
    use_bf16: bool,
    use_fp16: bool,
    on_cuda: bool,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_rows = len(ground_truth_df)

    # CO DINH random_state cho fold split (khong doi theo seed) - dung 1 phan
    # chia fold nhu nhau cho ca 5 lan chay, chi doi seed khoi tao/train.
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=FOLD_SPLIT_SEED)
    oof_proba = np.zeros((n_rows, len(VALID_LABELS)), dtype=np.float64)
    oof_pred = np.full(n_rows, -1, dtype=int)

    with tempfile.TemporaryDirectory(prefix=f"phobert_seedvar_{seed}_") as scratch:
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_rows), y), start=1):
            train_frame = ground_truth_df.iloc[train_idx].reset_index(drop=True)
            val_frame = ground_truth_df.iloc[val_idx].reset_index(drop=True)

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
                class_weights=class_weights,
            )
            trainer.train()

            probabilities = predict_pooled_probabilities(trainer, val_chunk_frame, val_dataset, len(val_frame))
            oof_proba[val_idx] = probabilities
            oof_pred[val_idx] = probabilities.argmax(axis=1)

            del trainer, model
            if on_cuda:
                torch.cuda.empty_cache()

    metrics_df = compute_metrics_table(y, oof_pred)
    overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
    per_class = metrics_df.set_index("metric_scope")["f1"]
    return {
        "seed": seed,
        "macro_f1": float(overall["f1"]),
        "accuracy": float(overall["accuracy"]),
        "f1_negative": float(per_class["negative"]),
        "f1_neutral": float(per_class["neutral"]),
        "f1_positive": float(per_class["positive"]),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = on_cuda and not args.no_amp and not args.fp16 and torch.cuda.is_bf16_supported()
    use_fp16 = on_cuda and not args.no_amp and not use_bf16
    print("Base model :", BASE_MODEL_PATH)
    print("Device     :", torch.cuda.get_device_name(0) if on_cuda else "cpu")
    print("Seeds      :", SEEDS, "| fold split seed (co dinh):", FOLD_SPLIT_SEED)

    ground_truth_df = load_ground_truth()
    ground_truth_df = ground_truth_df.assign(
        label=encode_labels(ground_truth_df["ground_truth_label"])
    ).reset_index(drop=True)
    y = np.asarray(ground_truth_df["label"], dtype=int)
    print("Documents  :", len(ground_truth_df))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    rows = []
    overall_start = time.perf_counter()
    for seed in SEEDS:
        seed_start = time.perf_counter()
        result = run_one_seed(seed, ground_truth_df, y, tokenizer, data_collator, args, use_bf16, use_fp16, on_cuda)
        rows.append(result)
        print(
            f"seed={seed}: macro_f1={result['macro_f1']:.4f} acc={result['accuracy']:.4f} "
            f"({time.perf_counter() - seed_start:.1f}s)"
        )

    per_seed_df = pd.DataFrame(rows)
    summary = {"model": "phobert_finetune"}
    for column in ("macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"):
        summary[f"{column}_mean"] = float(per_seed_df[column].mean())
        summary[f"{column}_std"] = float(per_seed_df[column].std(ddof=1))
    summary_df = pd.DataFrame([summary])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(DATA_DIR / "seed_variance_finetune_per_seed.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(DATA_DIR / "seed_variance_finetune_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Ket qua tung seed (5-fold CV, {len(ground_truth_df)} bai, fold split co dinh) ===")
    print(per_seed_df.to_string(index=False))
    print(f"\n=== Mean +/- std qua {len(SEEDS)} seed ===")
    print(summary_df.to_string(index=False))
    print(f"\nTong thoi gian: {time.perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
