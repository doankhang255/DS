"""Sắp xếp lại corpus theo publication_date đã sửa, đánh lại source_row_id = vị trí mới.

Mặc định chỉ kiểm tra (dry run). Thêm --apply để ghi: sao lưu mọi file sẽ sửa vào
data_news/backup_before_reorder_20261020/, ghi bảng ánh xạ cũ -> mới, hoán vị 11 parquet
thẳng hàng với corpus, đổi source_row_id trong các file nhãn / npz / parquet theo ID.
Mọi file mới đều tính từ bản sao lưu, nên lần chạy bị dừng giữa chừng chỉ cần chạy lại
--apply là làm tiếp đúng (file đã xong được ghi lại y hệt). Chưa có bản sao lưu mà corpus
đã sắp rồi thì tự dừng. review_extract_rows_5000_5499.csv không bị đụng tới: cột row trong
đó vẫn là số thứ tự cũ, tra sang số mới bằng source_row_id_map_20261020.csv.

    python EDA_clean/reorder_corpus_by_date.py            # chỉ kiểm tra
    python EDA_clean/reorder_corpus_by_date.py --apply    # ghi thật
"""
from __future__ import annotations

import glob
import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "data_news" / "backup_before_reorder_20261020"
MAP_PATH = ROOT / "data_news" / "source_row_id_map_20261020.csv"
CORPUS = "data_news/equity_news_clean_content.parquet"
BSL = "sentiment_label_method"
ALIGNED = [
    "data_news/data_tokenized/equity_news_tokenized_underthesea.parquet",
    "data_news/data_tokenized/equity_news_tokenized_vncorenlp.parquet",
    f"{BSL}/Lexicon_based/data/equity_news_content_sentiment_ratios.parquet",
    f"{BSL}/Lexicon_based/Scoring/data/article_scores.parquet",
    f"{BSL}/Lexicon_based/Scoring/data/article_scores_labeled.parquet",
    f"{BSL}/Lexicon_based/Scoring_Intensity/data/article_scores_intensity.parquet",
    f"{BSL}/Lexicon_based/Scoring_Intensity/data/article_scores_intensity_labeled.parquet",
    f"{BSL}/Traditional_ML/inference/data/article_scores_random_forest.parquet",
    f"{BSL}/Transfer_Learning/improve/data/production_clean_20261012/teacher_e3_article_scores.parquet",
    f"{BSL}/Transfer_Learning/inference/data/article_scores_phobert.parquet",
]
ID_CSVS = [
    "data_news/GT.csv",
    "data_news/Đức_GT.csv",
    "data_news/Đạt_GT.csv",
    f"{BSL}/data/bai_lech_ban_dat_annotator2.csv",
    f"{BSL}/data/inter_annotator_agreement_filtered.csv",
    f"{BSL}/Lexicon_based/Scoring/data/false_positive_ground_truth_detail.csv",
    f"{BSL}/Lexicon_based/Scoring/data/ground_truth_merged.csv",
    f"{BSL}/Transfer_Learning/improve/data/production_clean_20261012/distill_unlabeled_row_ids.csv",
]
ID_PARQUETS = [f"{BSL}/Traditional_ML/data/ground_truth_labeled_tokenized.parquet"]
NPZS = sorted(
    Path(p).relative_to(ROOT).as_posix()
    for p in glob.glob(str(ROOT / BSL / "Traditional_ML" / "data" / "*_oof_probabilities.npz"))
)
TARGETS = [CORPUS, *ALIGNED, *ID_CSVS, *ID_PARQUETS, *NPZS]
APPLY = "--apply" in sys.argv


def norm_text(s: pd.Series) -> pd.Series:
    s = pd.Series(s.to_numpy(dtype=object, na_value=""), dtype=object).astype(str)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def read_id_csv(path: Path) -> tuple[pd.DataFrame, str, str]:
    raw = path.read_bytes()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    terminator = "\r\n" if raw.count(b"\r\n") > 0 else "\n"
    df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
    return df, encoding, terminator


def csv_bytes(df: pd.DataFrame, encoding: str, terminator: str) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator=terminator)
    text = buf.getvalue()
    return (("﻿" + text) if encoding == "utf-8-sig" else text).encode("utf-8")


def replace_with_retry(tmp: Path, path: Path) -> None:
    # Windows: Defender / trình đánh chỉ mục / VS Code có thể giữ file đích vài giây -> thử lại
    for attempt in range(30):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(1)


def write_parquet(table: pa.Table, path: Path, like: Path) -> None:
    compression = pq.ParquetFile(like).metadata.row_group(0).column(0).compression.lower()
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression=compression)
    replace_with_retry(tmp, path)


def main() -> None:
    resume = BACKUP.exists()
    src = BACKUP if resume else ROOT   # nguồn = bản gốc (bản sao lưu nếu đã có)
    if resume:
        missing = [rel for rel in TARGETS if not (BACKUP / rel).exists()]
        assert not missing, f"Bản sao lưu thiếu file: {missing}"
        print(f"Đã có bản sao lưu {BACKUP}: tính lại mọi file từ bản sao lưu")
    aligned, id_csvs, id_parquets, npzs = (
        [rel for rel in group if (ROOT / rel).exists()] for group in (ALIGNED, ID_CSVS, ID_PARQUETS, NPZS)
    )
    targets = [CORPUS, *aligned, *id_csvs, *id_parquets, *npzs]
    skipped = [rel for rel in TARGETS if rel not in targets]
    if skipped:
        print("Bỏ qua file đã xóa (sẽ tạo lại khi chạy lại):", skipped)

    corpus = pq.read_table(src / CORPUS)
    n = corpus.num_rows
    ids = corpus["source_row_id"].to_numpy()
    dates = corpus["publication_date"].to_numpy()
    assert (ids == np.arange(n)).all(), "source_row_id gốc không phải 0..n-1"
    assert not np.isnat(dates).any(), "có publication_date rỗng"
    if (np.diff(dates.astype("int64")) >= 0).all():
        sys.exit("Corpus gốc đã sắp theo ngày rồi - không làm lại (tránh hoán vị hai lần).")

    order = np.argsort(dates, kind="stable")          # vị trí cũ của dòng mới thứ i
    new_of_old = np.empty(n, dtype=np.int64)
    new_of_old[order] = np.arange(n)
    moved = int((order != np.arange(n)).sum())
    print(f"Corpus {n} bài, {moved} bài đổi vị trí")

    corpus_link = corpus["link"].to_pandas()
    corpus_title = norm_text(corpus["title"].to_pandas())
    corpus_dates = pd.Series(dates)

    # Upstream: thứ tự gốc có trùng thứ tự trong equity_news.parquet không (để chạy lại
    # clean_content.py có sắp xếp ổn định thì ra đúng thứ tự này)
    eq_link = pd.read_parquet(ROOT / "data_news" / "equity_news.parquet", columns=["link"])["link"]
    if eq_link.is_unique and corpus_link.is_unique:
        pos = pd.Series(np.arange(len(eq_link)), index=eq_link.values).reindex(corpus_link.values)
        print("equity_news.parquet: link duy nhất; corpus là dãy con theo đúng thứ tự:",
              bool(pos.notna().all() and pos.is_monotonic_increasing))
    else:
        print("equity_news.parquet / corpus có link trùng:", (~eq_link.is_unique), (~corpus_link.is_unique))

    # 1. Kiểm tra các parquet thẳng hàng với corpus
    for rel in aligned:
        f = pq.ParquetFile(src / rel)
        names = f.schema_arrow.names
        assert f.metadata.num_rows == n, rel
        assert not any(c.startswith("__index_level") for c in names), rel
        cols = ["link"] + [c for c in ("publication_date", "source_row_id") if c in names]
        t = pq.read_table(src / rel, columns=cols).to_pandas()
        ok_link = bool((t["link"].values == corpus_link.values).all())
        ok_date = bool((t["publication_date"].values == corpus_dates.values).all()) if "publication_date" in t else None
        ok_id = bool((t["source_row_id"].values == np.arange(n)).all()) if "source_row_id" in t else None
        print(f"  thẳng hàng {rel.split('/')[-1]}: link={ok_link} date={ok_date} id={ok_id}")
        assert ok_link and ok_date is not False and ok_id is not False, rel

    # 2. Kiểm tra các file theo ID: ID hợp lệ, tiêu đề khớp corpus, ghi lại không đổi byte
    for rel in id_csvs:
        df, enc, term = read_id_csv(src / rel)
        old = df["source_row_id"].astype(int).to_numpy()
        assert ((old >= 0) & (old < n)).all(), rel
        same_bytes = csv_bytes(df, enc, term) == (src / rel).read_bytes()
        title_ok = None
        if "title" in df:
            title_ok = int((norm_text(df["title"]).values == corpus_title.values[old]).sum())
        date_ok = None
        if "publication_date" in df:
            d = pd.to_datetime(df["publication_date"], errors="coerce", format="ISO8601")
            date_ok = int((d.values == corpus_dates.values[old]).sum())
        print(f"  ID {rel.split('/')[-1]}: {len(df)} dòng, ghi lại giữ nguyên byte={same_bytes}, "
              f"title khớp={title_ok}, date khớp={date_ok}")
        assert same_bytes, rel
    for rel in id_parquets:
        t = pq.read_table(src / rel)
        old = t["source_row_id"].to_numpy()
        title_ok = None
        if "title" in t.schema.names:
            title_ok = int((norm_text(t["title"].to_pandas()).values == corpus_title.values[old]).sum())
        print(f"  ID {rel.split('/')[-1]}: {t.num_rows} dòng, title khớp={title_ok}")
    for rel in npzs:
        with np.load(src / rel) as z:
            old = z["source_row_id"]
            assert ((old >= 0) & (old < n)).all(), rel
            print(f"  ID {rel.split('/')[-1]}: {len(old)} id, keys={list(z.keys())}")

    if not APPLY:
        print("Dry run - chưa ghi gì. Chạy lại với --apply.")
        return

    # 3. Sao lưu (chỉ lần đầu)
    for rel in targets:  # file đang mở (Excel...) thì dừng trước khi ghi bất cứ gì
        with open(ROOT / rel, "r+b"):
            pass
    if not resume:
        for rel in targets:
            dst = BACKUP / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
        print(f"Đã sao lưu {len(targets)} file vào {BACKUP}")

    # 4. Bảng ánh xạ cũ -> mới
    pd.DataFrame({
        "old_source_row_id": np.arange(n),
        "new_source_row_id": new_of_old,
        "publication_date": corpus_dates.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "link": corpus_link.values,
    }).to_csv(MAP_PATH, index=False, encoding="utf-8-sig")

    # 5. Hoán vị corpus và 10 parquet thẳng hàng
    take = pa.array(order)
    for rel in [CORPUS, *aligned]:
        t = pq.read_table(BACKUP / rel).take(take)
        if "source_row_id" in t.schema.names:
            i = t.schema.get_field_index("source_row_id")
            t = t.set_column(i, t.schema.field(i), pa.array(np.arange(n), type=t.schema.field(i).type))
        write_parquet(t, ROOT / rel, BACKUP / rel)
        print("  đã sắp:", rel)

    # 6. Đổi ID trong các file theo ID (giữ nguyên thứ tự dòng)
    for rel in id_csvs:
        df, enc, term = read_id_csv(BACKUP / rel)
        df["source_row_id"] = new_of_old[df["source_row_id"].astype(int).to_numpy()].astype(str)
        tmp = (ROOT / rel).with_name(Path(rel).name + ".tmp")
        tmp.write_bytes(csv_bytes(df, enc, term))
        replace_with_retry(tmp, ROOT / rel)
        print("  đã đổi ID:", rel)
    for rel in id_parquets:
        t = pq.read_table(BACKUP / rel)
        i = t.schema.get_field_index("source_row_id")
        f = t.schema.field(i)
        t = t.set_column(i, f, pa.array(new_of_old[t["source_row_id"].to_numpy()], type=f.type))
        write_parquet(t, ROOT / rel, BACKUP / rel)
        print("  đã đổi ID:", rel)
    for rel in npzs:
        with np.load(BACKUP / rel) as z:
            arrays = {k: z[k] for k in z.keys()}
        arrays["source_row_id"] = new_of_old[arrays["source_row_id"]].astype(arrays["source_row_id"].dtype)
        tmp = (ROOT / rel).with_name(Path(rel).stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        replace_with_retry(tmp, ROOT / rel)
        print("  đã đổi ID:", rel)

    # 7. Kiểm tra sau khi ghi
    new = pq.read_table(ROOT / CORPUS)
    new_dates = new["publication_date"].to_numpy()
    new_link = new["link"].to_pandas()
    new_title = norm_text(new["title"].to_pandas())
    assert (np.diff(new_dates.astype("int64")) >= 0).all()
    assert (new["source_row_id"].to_numpy() == np.arange(n)).all()
    assert (new_link.values == corpus_link.values[order]).all()
    for rel in aligned:
        t = pq.read_table(ROOT / rel, columns=["link"]).to_pandas()
        assert (t["link"].values == new_link.values).all(), rel
    for rel in id_csvs:
        before, _, _ = read_id_csv(BACKUP / rel)
        after, _, _ = read_id_csv(ROOT / rel)
        old = before["source_row_id"].astype(int).to_numpy()
        nw = after["source_row_id"].astype(int).to_numpy()
        assert (new_link.values[nw] == corpus_link.values[old]).all(), rel
        assert before.drop(columns="source_row_id").equals(after.drop(columns="source_row_id")), rel
        if "title" in after:
            assert (new_title.values[nw] == corpus_title.values[old]).all(), rel
    for rel in npzs:
        with np.load(BACKUP / rel) as zb, np.load(ROOT / rel) as za:
            assert (new_link.values[za["source_row_id"]] == corpus_link.values[zb["source_row_id"]]).all(), rel
            assert all((zb[k] == za[k]).all() for k in zb.keys() if k != "source_row_id"), rel
    for rel in id_parquets:
        tb = pq.read_table(BACKUP / rel)
        ta = pq.read_table(ROOT / rel)
        assert (new_link.values[ta["source_row_id"].to_numpy()] == corpus_link.values[tb["source_row_id"].to_numpy()]).all(), rel
    print("Kiểm tra sau khi ghi: OK")


if __name__ == "__main__":
    main()
