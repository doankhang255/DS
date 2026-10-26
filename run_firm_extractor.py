"""Chạy extract_symbols_offline cho toàn bộ corpus (126.576 bài), xuất CSV theo chunk.

Mỗi chunk (mặc định 5.000 bài) ghi ra file tạm rồi mới đổi tên (os.replace), nên lần
chạy bị ngắt giữa chừng không để lại file hỏng; chạy lại thì bỏ qua các chunk đã có
(đổi extractor hay danh sách mã thì phải xoá thư mục chunks trước khi chạy lại).
Cuối cùng kiểm tra từng chunk đủ đúng các bài của nó rồi gộp thành 2 file:

- firm_symbols.csv: bài có ít nhất 1 mã. Cột giống first100_symbols_offline.csv:
  row (= source_row_id, số thứ tự dòng trong equity_news_clean_content.parquet),
  title, link, matched_by, evidence, symbol; mỗi mã 1 dòng.
- firm_unmatched.csv: bài bị loại, 1 bài 1 dòng, để người review: row, publication_date,
  title, link, reason, evidence, description. reason = "không khớp mã nào", hoặc bài nhắc
  tên dùng chung cho nhiều mã ("CTCP Nam Việt" = ANV/NAV) mà không ghi mã nào trong số
  đó (evidence ghi tên và các mã).

    python run_firm_extractor.py [--chunk-size 5000] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from call_firm_extractor import extract_symbols_offline

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data_news" / "equity_news_clean_content.parquet"
OUT_DIR = ROOT / "data_news" / "firm_extract"
COLUMNS = ["row", "title", "link", "matched_by", "evidence", "symbol"]


def chunk_path(chunk_dir: Path, start: int, end: int) -> Path:
    return chunk_dir / f"rows_{start:06d}_{end - 1:06d}.csv"


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def read_chunk(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: đọc lại đúng nguyên văn, không biến "NA"/"NULL" thành rỗng
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def process_chunk(articles: pd.DataFrame, path: Path) -> tuple[Path, int, int]:
    records = []
    for row, title, description, content, link in articles.itertuples(name=None):
        found = extract_symbols_offline(title, description, content)
        if not found:
            records.append({"row": row, "title": title, "link": link})
        for item in found:
            records.append({"row": row, "title": title, "link": link, **item})
    write_csv_atomic(pd.DataFrame(records, columns=COLUMNS), path)
    return path, len(articles), len(records)


def check_chunk(path: Path, start: int, end: int) -> pd.DataFrame:
    df = read_chunk(path)
    rows = set(df["row"].astype(int))
    if list(df.columns) != COLUMNS or rows != set(range(start, end)):
        raise ValueError(f"{path.name}: thiếu/thừa bài hoặc sai cột - xoá file này rồi chạy lại")
    return df


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N bài đầu (để thử)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    chunk_dir = args.out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(INPUT_PATH, columns=["title", "description", "content", "link"])
    corpus = corpus.reset_index(drop=True)
    if args.limit:
        corpus = corpus.head(args.limit)
    n = len(corpus)
    bounds = [(s, min(s + args.chunk_size, n)) for s in range(0, n, args.chunk_size)]
    todo = [(s, e) for s, e in bounds if not chunk_path(chunk_dir, s, e).exists()]
    print(f"{n:,} bài, {len(bounds)} chunk; đã có {len(bounds) - len(todo)}, cần chạy {len(todo)}", flush=True)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_chunk, corpus.iloc[s:e], chunk_path(chunk_dir, s, e)) for s, e in todo]
        for done, future in enumerate(as_completed(futures), 1):
            path, n_articles, n_rows = future.result()
            print(f"[{done}/{len(todo)}] {path.name}: {n_articles:,} bài -> {n_rows:,} dòng "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    merged = pd.concat([check_chunk(chunk_path(chunk_dir, s, e), s, e) for s, e in bounds], ignore_index=True)
    has_symbol = merged["symbol"].ne("")
    matched = merged[has_symbol]
    write_csv_atomic(matched, args.out_dir / "firm_symbols.csv")

    excluded = merged[~has_symbol]
    unmatched_rows = excluded["row"].astype(int)
    details = pd.read_parquet(INPUT_PATH, columns=["publication_date", "description"]).reset_index(drop=True)
    unmatched = pd.DataFrame({
        "row": unmatched_rows.to_numpy(),
        "publication_date": details["publication_date"].iloc[unmatched_rows].to_numpy(),
        "title": excluded["title"].to_numpy(),
        "link": excluded["link"].to_numpy(),
        "reason": excluded["matched_by"].replace("", "không khớp mã nào").to_numpy(),
        "evidence": excluded["evidence"].to_numpy(),
        "description": details["description"].iloc[unmatched_rows].to_numpy(),
    })
    write_csv_atomic(unmatched, args.out_dir / "firm_unmatched.csv")

    n_articles = merged["row"].nunique()
    print(f"\nĐã gộp {len(bounds)} chunk vào {args.out_dir}")
    print(f"firm_symbols.csv: {matched['row'].nunique():,}/{n_articles:,} bài có ít nhất 1 mã, "
          f"{len(matched):,} cặp bài-mã")
    print(matched["matched_by"].value_counts().to_string())
    print(f"firm_unmatched.csv: {len(unmatched):,} bài bị loại ({len(unmatched) / n_articles:.1%})")
    print(unmatched["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
