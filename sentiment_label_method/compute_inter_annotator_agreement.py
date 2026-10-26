"""Đo độ đồng thuận liên-chú-giải (inter-annotator agreement) giữa nhãn gốc
(`data_news/GT.csv`, annotator 1) và nhãn của annotator
2 (`data_news/Đức_GT.csv`) trên đúng những bài cả 2 người cùng gán -
theo góp ý của mentor (Mike Nguyen, 2026-09-18 mục 1):

    "nếu e k đủ tài nguyên, Kappa không cần gán lại toàn bộ ground truth. Lấy
    mẫu phân tầng 250-300 bài, người thứ hai gán độc lập, mù với nhãn cũ,
    xáo thứ tự để tránh anchoring. Vì ba lớp có thứ tự (neg < neu < pos),
    dùng weighted kappa hoặc Krippendorff ordinal alpha thay cho Cohen's
    kappa thường."

Dùng WEIGHTED KAPPA (linear + quadratic) vì 3 lớp CÓ THỨ TỰ
(Negative < Neutral < Positive) - nhầm Negative<->Positive phải bị phạt nặng
hơn nhầm Negative<->Neutral hay Neutral<->Positive, khác Cohen's kappa
thường (coi mọi cặp nhầm lẫn ngang nhau).

Output:
  - In ra raw agreement, Cohen kappa (không trọng số), weighted kappa
    (linear/quadratic), confusion matrix, và agreement RIÊNG theo từng lớp
    (theo gợi ý mục 3: "Khi có kappa, tính agreement riêng theo từng lớp").
  - Lưu data/inter_annotator_agreement_436.csv: đúng 436 bài đã LỌC (chỉ
    giữ bài có nhãn CẢ 2 người, loại 48 bài annotator_2 gán thêm nhưng
    không có trong GT.csv) - kèm title, 2 nhãn, và cờ
    agree (True/False) - để xem lại/dùng tiếp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NEWS_DIR = PROJECT_ROOT / "data_news"

GROUND_TRUTH_PATH = DATA_NEWS_DIR / "GT.csv"
ANNOTATOR_2_PATH = DATA_NEWS_DIR / "Đức_GT.csv"
OUTPUT_FILTERED_PATH = Path(__file__).resolve().parent / "data" / "inter_annotator_agreement_filtered.csv"

LABELS = ["Negative", "Neutral", "Positive"]
ORDER_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def load_labels(path: Path, rename_to: str, with_title: bool = False) -> pd.DataFrame:
    columns = ["source_row_id", "sentiment", "title"] if with_title else ["source_row_id", "sentiment"]
    df = pd.read_csv(path, encoding="utf-8-sig")[columns].copy()
    df["source_row_id"] = df["source_row_id"].astype(int)
    return df.rename(columns={"sentiment": rename_to})


def per_class_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        # Agreement riêng theo từng lớp: trong số bài mà NHÃN GỐC là `label`,
        # bao nhiêu % annotator_2 cũng gán đúng `label` đó.
        subset = merged.loc[merged["sentiment_gt"] == label]
        if len(subset) == 0:
            rows.append({"label": label, "n": 0, "agreement": float("nan")})
            continue
        agree = (subset["sentiment_gt"] == subset["sentiment_ann2"]).mean()
        rows.append({"label": label, "n": len(subset), "agreement": agree})
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ann2 = load_labels(ANNOTATOR_2_PATH, "sentiment_ann2", with_title=True)
    gt = load_labels(GROUND_TRUTH_PATH, "sentiment_gt")
    print(f"Đức_GT.csv: {len(ann2)} bài")
    print(f"GT.csv: {len(gt)} bài")

    merged = ann2.merge(gt, on="source_row_id", how="inner")
    missing_in_gt = len(ann2) - len(merged)
    print(f"\nSố bài có nhãn CẢ 2 người (dùng để tính agreement): {len(merged)}")
    if missing_in_gt:
        print(
            f"CẢNH BÁO: {missing_in_gt} bài trong Đức_GT.csv KHÔNG khớp source_row_id "
            f"nào trong GT.csv (annotator 2 gán thêm bài mới chưa có nhãn gốc?) "
            f"- các bài này bị loại khỏi phép đo agreement."
        )

    y_gt = merged["sentiment_gt"].map(ORDER_MAP)
    y_ann2 = merged["sentiment_ann2"].map(ORDER_MAP)

    raw_agreement = (merged["sentiment_gt"] == merged["sentiment_ann2"]).mean()
    kappa_plain = cohen_kappa_score(y_gt, y_ann2)
    kappa_linear = cohen_kappa_score(y_gt, y_ann2, weights="linear")
    kappa_quadratic = cohen_kappa_score(y_gt, y_ann2, weights="quadratic")

    print(f"\n=== KẾT QUẢ (n={len(merged)}) ===")
    print(f"Raw agreement (khớp nguyên văn):    {raw_agreement * 100:.2f}%")
    print(f"Cohen's kappa (không trọng số):     {kappa_plain:.4f}")
    print(f"Weighted kappa (linear):            {kappa_linear:.4f}")
    print(f"Weighted kappa (quadratic):         {kappa_quadratic:.4f}  <- dùng số này (mentor khuyến nghị, 3 lớp có thứ tự)")

    cm = confusion_matrix(merged["sentiment_gt"], merged["sentiment_ann2"], labels=LABELS)
    print("\nConfusion matrix (hàng=nhãn gốc/annotator 1, cột=annotator 2):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    print("\nAgreement RIÊNG theo từng lớp (trong số bài nhãn gốc = lớp đó, annotator_2 đồng ý bao nhiêu %):")
    print(per_class_agreement(merged).to_string(index=False))

    filtered = merged.copy()
    filtered["agree"] = filtered["sentiment_gt"] == filtered["sentiment_ann2"]
    filtered = filtered[["source_row_id", "title", "sentiment_gt", "sentiment_ann2", "agree"]]
    OUTPUT_FILTERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_FILTERED_PATH, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu {len(filtered)} bài đã lọc (dùng để tính kappa) vào: {OUTPUT_FILTERED_PATH}")


if __name__ == "__main__":
    main()
