"""Phần dùng chung cho mọi crawler trang báo: lưu parquet (không dùng json), phase meta cho trang chỉ
có danh sách phân trang (crawl_paged_meta), phase content (crawl_content).

Parquet không ghi nối được, nên trong lúc crawl các dòng được gom lại rồi ghi thành nhiều file
nhỏ (part-*.parquet) trong 1 thư mục: bị ngắt giữa chừng chỉ mất phần chưa ghi (tối đa
flush_every dòng), lần sau tải lại. Các máy chạy --shard ghi part tên khác nhau (giờ + pid), chép
chung vào 1 thư mục là gộp được. Cuối mỗi phase, các part được gộp (khử trùng theo link) thành 1
file parquet + 1 file csv 1000 bài ngẫu nhiên để kiểm tra bằng mắt.

Thư mục của 1 trang báo (Crawler/data/<trang>/):
    meta_parts/        part-*.parquet - cột lấy từ trang danh sách
    article_parts/     part-*.parquet - bài đầy đủ (có content)
    windows.parquet    (kênh, tháng) đã liệt kê xong
    failed.parquet     bài lỗi vĩnh viễn (404...), không thử lại
    meta.parquet, news.parquet                     - file gộp
    meta_sample_1000.csv, news_sample_1000.csv     - mẫu ngẫu nhiên để kiểm tra
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

SAMPLE_SIZE = 1000
WINDOW_COLUMNS = ["channel_id", "month", "articles"]
FAILED_COLUMNS = ["link", "reason"]


class PartWriter:
    """Gom dòng rồi ghi thành part-*.parquet trong `folder` (thay cho file jsonl ghi nối)."""

    def __init__(self, folder: Path, columns: list[str], flush_every: int = 500) -> None:
        self.folder = folder
        self.columns = columns
        self.flush_every = flush_every
        self.rows: list[dict] = []
        self.count = 0
        folder.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.count += 1
        name = f"part-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{self.count:05d}.parquet"
        frame = pd.DataFrame(self.rows, columns=self.columns).astype("string")
        tmp = self.folder / (name + ".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, self.folder / name)  # không để lại part ghi dở
        self.rows = []


def read_parts(folder: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Gộp mọi part trong thư mục (chưa khử trùng). `columns` để chỉ đọc vài cột cho nhanh."""
    paths = sorted(folder.glob("part-*.parquet")) if folder.exists() else []
    if not paths:
        return pd.DataFrame(columns=columns or [])
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class SiteStore:
    """Toàn bộ file dữ liệu của 1 trang báo trong Crawler/data/<trang>/."""

    def __init__(self, folder: Path, meta_columns: list[str], article_columns: list[str]) -> None:
        self.folder = folder
        self.meta_columns = meta_columns
        self.article_columns = article_columns
        self.meta_parts = folder / "meta_parts"
        self.article_parts = folder / "article_parts"
        self.windows_path = folder / "windows.parquet"
        self.failed_path = folder / "failed.parquet"
        self.meta_path = folder / "meta.parquet"
        self.news_path = folder / "news.parquet"

    # --- bảng nhỏ: (kênh, tháng) đã xong, bài lỗi vĩnh viễn ---
    def _read_table(self, path: Path, columns: list[str]) -> pd.DataFrame:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)

    def _append_table(self, path: Path, rows: list[dict], columns: list[str]) -> None:
        if rows:
            old = self._read_table(path, columns)
            write_parquet(pd.concat([old, pd.DataFrame(rows, columns=columns)], ignore_index=True).astype("string"), path)

    def done_windows(self) -> set[tuple[str, str]]:
        """{(mã kênh dạng chuỗi, "YYYY-MM")}."""
        table = self._read_table(self.windows_path, WINDOW_COLUMNS)
        return {(str(channel), str(month)) for channel, month in zip(table["channel_id"], table["month"])}

    def mark_windows(self, rows: list[dict]) -> None:
        """rows: {"channel_id", "month", "articles"}. Gọi SAU KHI đã flush meta của các tháng đó."""
        self._append_table(self.windows_path, rows, WINDOW_COLUMNS)

    def failed_links(self) -> set[str]:
        return set(self._read_table(self.failed_path, FAILED_COLUMNS)["link"])

    def add_failed(self, rows: list[dict]) -> None:
        self._append_table(self.failed_path, rows, FAILED_COLUMNS)

    # --- meta / bài ---
    def meta_writer(self) -> PartWriter:
        return PartWriter(self.meta_parts, self.meta_columns, flush_every=5000)

    def article_writer(self) -> PartWriter:
        return PartWriter(self.article_parts, self.article_columns, flush_every=300)

    def known_meta_links(self) -> set[str]:
        return set(read_parts(self.meta_parts, ["link"])["link"])

    def read_meta(self) -> pd.DataFrame:
        return read_parts(self.meta_parts).reindex(columns=self.meta_columns).drop_duplicates(subset=["link"])

    def done_article_links(self) -> set[str]:
        return set(read_parts(self.article_parts, ["link"])["link"])

    def export(self, kind: str) -> int:
        """Gộp part -> meta.parquet / news.parquet + <kind>_sample_1000.csv (bài ngẫu nhiên, để kiểm tra)."""
        folder, columns, out_path = ((self.meta_parts, self.meta_columns, self.meta_path) if kind == "meta"
                                     else (self.article_parts, self.article_columns, self.news_path))
        frame = read_parts(folder).reindex(columns=columns).astype("string")
        frame = frame.mask(frame.eq("")).drop_duplicates(subset=["link"]).sort_values("publication_date", kind="mergesort")
        frame = frame.reset_index(drop=True)
        write_parquet(frame, out_path)
        sample_path = self.folder / f"{kind}_sample_{SAMPLE_SIZE}.csv"
        sample = frame.sample(min(SAMPLE_SIZE, len(frame)), random_state=0).sort_values("publication_date", kind="mergesort")
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel đọc đúng tiếng Việt
        print(f"[{kind}] tong {len(frame)} bai -> {out_path} | mau {len(sample)} bai -> {sample_path}")
        return len(frame)

    # --- chuyển dữ liệu của bản crawler cũ (.jsonl ở Crawler/data/) ---
    def migrate_legacy(self, prefix: str) -> None:
        """<prefix>_meta.jsonl, _windows.jsonl, _failed.jsonl, _articles*.jsonl -> parquet; file cũ
        (cả .parquet gộp cũ) cất vào Crawler/data/legacy/, không xoá."""
        data_dir = self.folder.parent
        legacy_files = sorted(data_dir.glob(f"{prefix}_*.jsonl"))
        if not legacy_files:
            return
        newest = max(path.stat().st_mtime for path in legacy_files)
        if time.time() - newest < 120:
            sys.exit(f"[chuyen doi] {data_dir}/{prefix}_*.jsonl vua duoc ghi < 2 phut truoc - co the crawler ban cu "
                     f"van dang chay. Cho no chay xong roi chay lai.")
        legacy = data_dir / "legacy"
        legacy.mkdir(exist_ok=True)
        for path in legacy_files:
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if path.name == f"{prefix}_windows.jsonl":
                self.mark_windows(rows)
            elif path.name == f"{prefix}_failed.jsonl":
                self.add_failed(rows)
            else:
                folder, columns = ((self.meta_parts, self.meta_columns) if path.name == f"{prefix}_meta.jsonl"
                                   else (self.article_parts, self.article_columns))
                writer = PartWriter(folder, columns, flush_every=len(rows) + 1)
                for row in rows:
                    writer.add(row)
                writer.flush()
            shutil.move(str(path), legacy / path.name)
            print(f"[chuyen doi] {path.name}: {len(rows)} dong -> parquet; file cu cat vao {legacy}")
        for path in data_dir.glob(f"{prefix}_*.parquet"):
            shutil.move(str(path), legacy / path.name)


FetchArticle = Callable[[object, dict], tuple[str, "dict | None", str, bool]]
# fetch_page(fetcher, mã kênh, trang) -> các bài trên trang theo thứ tự mới -> cũ, mỗi bài ít nhất có
# link, publication_date ("YYYY-MM-DD ..."); None nếu không tải được; [] nếu đã quá trang cuối
FetchPage = Callable[[object, str, int], "list[dict] | None"]


def _months_between(start: date, end: date) -> list[str]:
    return [f"{y}-{m:02d}" for y in range(start.year, end.year + 1) for m in range(1, 13)
            if f"{start:%Y-%m}" <= f"{y}-{m:02d}" <= f"{end:%Y-%m}"]


def _full_past_month(month: str, start: date, end: date) -> bool:
    """Tháng nằm trọn trong [start, end] và đã qua (tháng hiện tại còn bài mới)."""
    year, mon = map(int, month.split("-"))
    first, last = date(year, mon, 1), date(year, mon, calendar.monthrange(year, mon)[1])
    return start <= first and last <= end and last < date.today()


def _first_page_until(fetcher, fetch_page: FetchPage, channel: str, end: str, last_page_guess: int) -> int:
    """Trang nhỏ nhất có bài đăng <= end (hoặc đã quá trang cuối), tìm nhị phân. So bằng bài CŨ NHẤT
    trên trang, để 1 bài nổi bật / quảng bá chèn lạc vào danh sách không làm sai."""
    low, high = 1, last_page_guess
    while low < high:
        middle = (low + high) // 2
        items = fetch_page(fetcher, channel, middle)
        if items is None:
            raise RuntimeError(f"khong tai duoc trang {middle} cua kenh {channel}")
        if not items or min(item["publication_date"] for item in items) <= end:
            high = middle
        else:
            low = middle + 1
    return low


def crawl_paged_meta(args: argparse.Namespace, fetcher, store: SiteStore, channels: dict[str, str],
                     fetch_page: FetchPage, last_page_guess: int = 20000) -> None:
    """Phase meta cho trang báo chỉ có danh sách phân trang (mới -> cũ), không lọc được theo ngày.

    Với mỗi kênh: tìm nhị phân trang đầu tiên có bài <= --end, rồi lật dần sang trang cũ hơn tới khi
    gặp bài < --start. (kênh, tháng) đã đi qua trọn vẹn thì ghi vào windows.parquet; lần sau bắt
    đầu từ tháng mới nhất chưa xong và dừng khi các tháng còn lại đều đã xong."""
    known = store.known_meta_links()
    done = store.done_windows()
    print(f"[meta] {args.start} .. {args.end} | kenh {list(channels)} | da co {len(known)} bai, {len(done)} (kenh, thang) xong")
    writer = store.meta_writer()
    months = _months_between(args.start, args.end)
    new_rows = 0
    try:
        for channel, category in channels.items():
            pending = [m for m in months if (channel, m) not in done]
            if not pending:
                print(f"[meta] {channel}: moi thang da xong")
                continue
            # Bắt đầu từ cuối tháng mới nhất chưa xong (các tháng mới hơn đã liệt kê ở lần trước)
            year, mon = map(int, pending[-1].split("-"))
            top = min(args.end, date(year, mon, calendar.monthrange(year, mon)[1]))
            end_text, start_text = f"{top} 23:59:59", f"{args.start}"
            page = _first_page_until(fetcher, fetch_page, channel, end_text, last_page_guess)
            print(f"[meta] {channel} ({category}): {args.start} .. {top}, bat dau tu trang {page}")

            current, month_count = pending[-1], 0

            def close_month() -> None:
                """Đã đi qua hết tháng `current` -> ghi meta ra đĩa rồi đánh dấu xong. Tháng 0 bài
                không đánh dấu (có thể do đọc trang lỗi), lần sau liệt kê lại."""
                writer.flush()
                if (month_count and _full_past_month(current, args.start, args.end)
                        and (channel, current) not in done):
                    store.mark_windows([{"channel_id": channel, "month": current, "articles": month_count}])
                    done.add((channel, current))
                print(f"  {current}: {month_count} bai (den trang {page})")

            stop = False
            items: list[dict] | None = []
            while not stop:
                items = fetch_page(fetcher, channel, page)
                if not items:  # None: không tải được; []: quá trang cuối
                    break
                for item in items:
                    published = item["publication_date"]
                    if published > end_text:
                        continue
                    if published < start_text:
                        stop = True
                        break
                    month = published[:7]
                    if month < current:
                        close_month()
                        current, month_count = month, 0
                        # Các tháng từ đây về --start đều đã xong ở lần chạy trước -> dừng sớm
                        if all((channel, m) in done for m in months if m <= current):
                            stop = True
                            break
                    month_count += 1
                    if item["link"] in known:
                        continue
                    writer.add({**item, "channel_id": channel, "category": category, "month": month})
                    known.add(item["link"])
                    new_rows += 1
                page += 1
            if items is None:
                # Tháng đang dở chưa đánh dấu xong -> lần sau liệt kê lại từ tháng này
                writer.flush()
                print(f"  [dung] khong tai duoc trang {page}, chay lai de lam tiep")
            else:
                close_month()
    finally:
        writer.flush()
    print(f"[meta] bai moi lan nay: {new_rows}")


def crawl_content(args: argparse.Namespace, fetcher, store: SiteStore, fetch_article: FetchArticle) -> None:
    """Phase content: tải trang từng bài trong meta (trong --start/--end, --channels, --shard) chưa có.

    fetch_article(fetcher, meta_row) -> (link, bài | None, lỗi, lỗi vĩnh viễn?)."""
    meta = store.read_meta().astype("string")
    first_month, last_month = f"{args.start:%Y-%m}", f"{args.end:%Y-%m}"
    in_range = meta[(meta["month"] >= first_month) & (meta["month"] <= last_month)] if len(meta) else meta
    if in_range.empty:
        print(f"[content] meta chua co bai nao trong {first_month} .. {last_month} ({store.meta_parts}). "
              f"Chay --phase meta cho khoang nay truoc.")
        return
    done_links = store.done_article_links()
    failed_links = store.failed_links()
    channels = {str(channel) for channel in args.channels}
    todo = in_range[in_range["channel_id"].astype(str).isin(channels)
                    & ~in_range["link"].isin(done_links) & ~in_range["link"].isin(failed_links)]
    if args.shard:
        shard, shards = map(int, args.shard.split("/"))
        todo = todo[pd.to_numeric(todo["article_id"], errors="coerce").fillna(0).astype("int64") % shards == shard - 1]
    if args.max_articles is not None:
        todo = todo.head(args.max_articles)
    rows = todo.to_dict("records")
    print(f"[content] {first_month} .. {last_month} | {len(meta)} bai trong meta, da tai {len(done_links)}, "
          f"loi bo qua {len(failed_links)} | can tai {len(rows)}")

    writer = store.article_writer()
    new_articles = 0
    failed: list[tuple[str, str]] = []
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futures = [pool.submit(fetch_article, fetcher, row) for row in rows]
        for count, future in enumerate(as_completed(futures), 1):
            url, row, error, permanent = future.result()
            if row is None:
                failed.append((url, error))
                if permanent:
                    store.add_failed([{"link": url, "reason": error}])
            else:
                writer.add(row)
                new_articles += 1
            if count % 200 == 0 or count == len(rows):
                rate = count / (time.monotonic() - started)
                eta_hours = (len(rows) - count) / rate / 3600
                print(f"  {count}/{len(rows)} bai | {rate:.2f} bai/giay | con ~{eta_hours:.1f} gio | loi {len(failed)}")
    finally:
        # Ctrl+C: huỷ các bài chưa tải thay vì chờ tải hết hàng đợi, và ghi nốt bài đã tải
        pool.shutdown(wait=True, cancel_futures=True)
        writer.flush()

    print(f"[content] bai moi lan nay: {new_articles} | loi/bo qua: {len(failed)}")
    for url, error in failed[:10]:
        print(f"  {error}: {url}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase", choices=["all", "meta", "content", "export"], default="all",
                        help="meta: cot tu trang danh sach; content: tai trang bai; export: chi gop part -> parquet + csv mau")
    parser.add_argument("--shard", default=None, help="K/N: may thu K trong N may chia nhau phase content (VD 1/3)")
    parser.add_argument("--max-articles", type=int, default=None, help="phase content: dung sau N bai (de chay thu)")
    parser.add_argument("--delay", type=float, default=1.0, help="khoang cach toi thieu (giay) giua 2 request")
    parser.add_argument("--workers", type=int, default=1, help="so luong request song song (van chiu --delay chung)")


def run_phases(args: argparse.Namespace, fetcher, store: SiteStore, crawl_meta, fetch_article: FetchArticle) -> None:
    if args.phase in ("all", "meta"):
        crawl_meta(args, fetcher)
        store.export("meta")
    if args.phase in ("all", "content"):
        crawl_content(args, fetcher, store, fetch_article)
    if args.phase in ("all", "content", "export"):
        store.export("news")
    if args.phase == "export":
        store.export("meta")
