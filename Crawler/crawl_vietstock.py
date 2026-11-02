"""Crawl tin vietstock.vn, xuất 2 file:
  - vietstock_dataset_news.parquet: 9 cột của dataset_news.parquet (3T) + content
    (link, domain, category, title, description, keywords, author, publication_date, year, content)
  - vietstock_equity_news_clean_content.parquet: đúng 9 cột của equity_news_clean_content.parquet
    (source_row_id, link, publication_date, domain_norm, category, title, description,
    keywords_norm, content), làm sạch content bằng chính các hàm của EDA_clean/clean_content.py

Cách lấy danh sách bài: endpoint ajax của chính trang (POST /StartPage/ChannelContentPage
với channelID, page, fromdate, todate; 10 bài/trang). Các kênh cha (chứng khoán, doanh
nghiệp, nhận định - phân tích, kinh tế, tài chính) gộp luôn bài của kênh con, nên chỉ cần
quét 5 kênh cha theo từng tháng. `category` lấy từ nhãn kênh con hiện trên từng dòng của
danh sách (VD "Chính sách", "Cổ phiếu"). Mặc định chỉ tải bài có category nằm trong
data_news/categories.csv (tin doanh nghiệp), thêm --all-categories để tải hết. Thông tin bài
lấy từ thẻ meta của trang bài, content lấy từ các đoạn <p> trong div#vst_detail (bỏ tiêu đề,
sapo, tác giả, nguồn).

Ví dụ (từ thư mục gốc dự án):
    .venv\\Scripts\\python.exe Crawler\\crawl_vietstock.py --start 2012-03-01 --end 2012-03-31 --max-articles 20
    .venv\\Scripts\\python.exe Crawler\\crawl_vietstock.py            # 2010-01-01 .. hôm nay, chạy tiếp được
    .venv\\Scripts\\python.exe Crawler\\crawl_vietstock.py --watch-hours 24   # xong thì mỗi ngày cập nhật bài mới
    .venv\\Scripts\\python.exe Crawler\\crawl_vietstock.py --export-only      # chỉ xuất lại 2 file parquet
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from lxml import html as lxml_html

BASE_URL = "https://vietstock.vn"
LIST_ENDPOINT = f"{BASE_URL}/StartPage/ChannelContentPage"
DOMAIN = "vietstock.vn"
USER_AGENT = "Mozilla/5.0 (research crawler; contact: doankhangll255@gmail.com)"

PARENT_CHANNELS = {
    144: "Chứng khoán",
    733: "Doanh nghiệp",
    579: "Nhận định - Phân tích",
    5307: "Kinh tế",
    734: "Tài chính",
}
PAGE_SIZE = 10

OUTPUT_COLUMNS = [
    "link",
    "domain",
    "category",
    "title",
    "description",
    "keywords",
    "author",
    "publication_date",
    "year",
    "content",
]
CLEAN_COLUMNS = [
    "source_row_id",
    "link",
    "publication_date",
    "domain_norm",
    "category",
    "title",
    "description",
    "keywords_norm",
    "content",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CATEGORIES_PATH = PROJECT_ROOT / "data_news" / "categories.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
JSONL_PATH = OUTPUT_DIR / "vietstock_articles.jsonl"
PARQUET_PATH = OUTPUT_DIR / "vietstock_dataset_news.parquet"
CLEAN_PARQUET_PATH = OUTPUT_DIR / "vietstock_equity_news_clean_content.parquet"

# Các đoạn trong div#vst_detail không thuộc thân bài
NON_BODY_CLASSES = {"pTitle", "pHead", "pAuthor", "pSource"}

ARTICLE_ID_PATTERN = re.compile(r"-(\d+)\.htm$")
TITLE_SUFFIX_PATTERN = re.compile(r"\s*\|\s*Vietstock\s*$", re.IGNORECASE)
TOTAL_ROWS_PATTERN = re.compile(r"totalChannelRow\s*=\s*(\d+)")
TIMEZONE_SUFFIX_PATTERN = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


class Fetcher:
    def __init__(self, min_interval: float, timeout: float = 30.0, retries: int = 4) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self.min_interval
        if start > now:
            time.sleep(start - now)

    def request(self, method: str, url: str, **kwargs) -> requests.Response | None:
        for attempt in range(self.retries):
            self._wait_for_slot()
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as error:
                print(f"  [loi mang] {url} ({error.__class__.__name__}), thu lai {attempt + 1}/{self.retries}")
                time.sleep(2 ** attempt * 2)
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                print(f"  [http {response.status_code}] {url}, thu lai {attempt + 1}/{self.retries}")
                time.sleep(2 ** attempt * 5)
                continue
            return response
        return None


def iter_month_windows(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        first = date(year, month, 1)
        last = date(year, month, calendar.monthrange(year, month)[1])
        yield max(first, start), min(last, end)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def parse_list_fragment(fragment: str) -> tuple[list[tuple[str, str]], int]:
    total_match = TOTAL_ROWS_PATTERN.search(fragment)
    total_rows = int(total_match.group(1)) if total_match else 0
    if not fragment.strip():
        return [], total_rows

    document = lxml_html.fromstring(f"<div>{fragment}</div>")
    items: list[tuple[str, str]] = []
    for node in document.xpath('//div[contains(@class, "channelContent")]'):
        hrefs = node.xpath('.//div[contains(@class, "post_img")]//a/@href')
        if not hrefs:
            continue
        categories = node.xpath('.//div[contains(@class, "meta3")]/a[1]/@title')
        items.append((urljoin(BASE_URL, hrefs[0]), categories[0].strip() if categories else ""))
    return items, total_rows


def load_allowed_categories() -> set[str]:
    categories = pd.read_csv(CATEGORIES_PATH, encoding="utf-8-sig")["category"]
    return {" ".join(str(value).split()).lower() for value in categories.dropna()}


def list_month_articles(
    fetcher: Fetcher,
    window_start: date,
    window_end: date,
    limit: int | None,
    allowed_categories: set[str] | None,
) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for channel_id, channel_name in PARENT_CHANNELS.items():
        page = 1
        while True:
            response = fetcher.request(
                "POST",
                LIST_ENDPOINT,
                data={
                    "channelID": channel_id,
                    "page": page,
                    "fromdate": window_start.isoformat(),
                    "todate": window_end.isoformat(),
                },
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            if response is None or response.status_code != 200:
                print(f"  [bo qua] kenh {channel_name} trang {page}: khong lay duoc danh sach")
                break
            items, total_rows = parse_list_fragment(response.text)
            for url, category in items:
                if allowed_categories is not None and " ".join(category.split()).lower() not in allowed_categories:
                    continue
                id_match = ARTICLE_ID_PATTERN.search(url)
                if id_match:
                    found.setdefault(id_match.group(1), (url, category))
            if not items or page >= math.ceil(total_rows / PAGE_SIZE):
                break
            if limit is not None and len(found) >= limit:
                break
            page += 1
        if limit is not None and len(found) >= limit:
            break
    return found


def first_attribute(document, xpath: str) -> str:
    values = document.xpath(xpath)
    return html.unescape(values[0]).strip() if values else ""


def extract_content(document) -> str:
    """Mỗi đoạn <p> của thân bài thành 1 dòng; bảng/ảnh bỏ qua."""
    containers = document.xpath('//div[@id="vst_detail"]')
    if not containers:
        return ""
    lines = []
    for paragraph in containers[0].xpath("./p"):
        if NON_BODY_CLASSES.intersection((paragraph.get("class") or "").split()):
            continue
        text = " ".join(paragraph.text_content().split())
        if text:
            lines.append(text)
    return "\n".join(lines)


def parse_article(content: bytes, url: str, list_category: str) -> dict | None:
    parser = lxml_html.HTMLParser(encoding="utf-8")
    document = lxml_html.fromstring(content, parser=parser)

    published_raw = first_attribute(document, '//meta[@property="article:published_time"]/@content')
    if not published_raw:
        published_raw = first_attribute(document, '//meta[@name="pubdate"]/@content')
    if not published_raw:
        return None
    publication_date = TIMEZONE_SUFFIX_PATTERN.sub("", published_raw).replace("T", " ")

    title = first_attribute(document, "//title/text()")
    if not title:
        title = first_attribute(document, '//meta[@property="og:title"]/@content')
    title = TITLE_SUFFIX_PATTERN.sub("", title)

    description = first_attribute(document, '//meta[@name="description"]/@content')
    if not description:
        description = first_attribute(document, '//meta[@property="og:description"]/@content')
    if description == title:
        description = ""

    category = list_category or first_attribute(document, '//meta[@property="article:section"]/@content')

    tags = {}
    for anchor in document.xpath('//ul[li[contains(@class, "tag_list")]]/li/a[starts-with(@href, "/tag/")]'):
        tag_name = anchor.text_content().strip()
        if tag_name:
            tags[tag_name] = anchor.get("href")

    author_nodes = document.xpath('//p[contains(@class, "pAuthor")]')
    author = " ".join(author_nodes[0].text_content().split()) if author_nodes else ""

    return {
        "link": url,
        "domain": DOMAIN,
        "category": category,
        "title": title,
        "description": description,
        "keywords": json.dumps(tags, ensure_ascii=False) if tags else "",
        "author": author,
        "publication_date": publication_date,
        "year": publication_date[:4],
        "content": extract_content(document),
    }


def fetch_article(fetcher: Fetcher, url: str, list_category: str) -> tuple[str, dict | None, str]:
    response = fetcher.request("GET", url)
    if response is None:
        return url, None, "khong tai duoc"
    if response.status_code != 200:
        return url, None, f"http {response.status_code}"
    row = parse_article(response.content, url, list_category)
    if row is None:
        return url, None, "khong co thoi gian dang"
    return url, row, ""


def load_done_links(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["link"])
    return done


def export_parquet(jsonl_path: Path, parquet_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).astype("string")
    result = result.mask(result.eq("")).drop_duplicates(subset=["link"]).sort_values("publication_date", kind="mergesort")
    result = result.reset_index(drop=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(parquet_path, index=False)
    return result


def keywords_to_norm(raw: object) -> object:
    # Giống news_clean.parse_dict_like_keywords: {"tag": "/tag/..."} -> "tag1, tag2"
    if pd.isna(raw) or not str(raw).strip():
        return pd.NA
    keys = [key for key in json.loads(raw) if key]
    return ", ".join(keys) if keys else pd.NA


def export_clean_parquet(raw: pd.DataFrame, parquet_path: Path) -> int:
    """Đưa bản crawl về đúng format equity_news_clean_content.parquet."""
    from EDA_clean.EDA_raw_news import count_words_in_text, normalize_text_columns
    from EDA_clean.clean_content import (
        keep_content_before_interest_marker,
        keep_content_between_dic_and_nav,
        remove_records_by_content1,
        remove_title_description_from_content,
        replace_short_content_with_description,
    )

    out = normalize_text_columns(raw, ["link", "category", "title", "description", "keywords"])
    out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce").dt.floor("min")
    out = out.loc[out["publication_date"].notna() & out["publication_date"].dt.year.ge(2010)]
    out = out.loc[out["description"].apply(count_words_in_text).ge(5)]
    out = out.loc[out["content"].notna() | out["description"].notna()].copy()
    out["domain_norm"] = pd.Series(DOMAIN, index=out.index, dtype="string")
    out["keywords_norm"] = out["keywords"].apply(keywords_to_norm).astype("string")
    out["content"] = out["content"].fillna("").astype(str)

    out, _ = remove_records_by_content1(out)
    out, _ = keep_content_between_dic_and_nav(out)
    out, _ = keep_content_before_interest_marker(out)
    out, _ = remove_title_description_from_content(out)
    out, _ = replace_short_content_with_description(out)

    out = out.sort_values("publication_date", kind="mergesort").reset_index(drop=True)
    out.insert(0, "source_row_id", range(len(out)))
    out = out[CLEAN_COLUMNS]
    out["content"] = out["content"].astype(str)
    out.to_parquet(parquet_path, index=False)
    return len(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="mac dinh: hom nay")
    parser.add_argument("--max-articles", type=int, default=None, help="dung sau khi lay du N bai moi (de chay thu)")
    parser.add_argument("--delay", type=float, default=0.5, help="khoang cach toi thieu (giay) giua 2 request")
    parser.add_argument("--workers", type=int, default=4, help="so luong tai bai song song (van chiu --delay chung)")
    parser.add_argument("--all-categories", action="store_true", help="tai ca bai ngoai data_news/categories.csv")
    parser.add_argument("--export-only", action="store_true", help="khong crawl, chi xuat lai parquet tu jsonl")
    parser.add_argument(
        "--watch-hours",
        type=float,
        default=0,
        help="sau khi crawl xong, cu moi N gio crawl lai 3 ngay gan nhat (0 = chay 1 lan roi thoat)",
    )
    return parser.parse_args()


def crawl_range(
    fetcher: Fetcher,
    start: date,
    end: date,
    done_links: set[str],
    allowed_categories: set[str] | None,
    workers: int,
    max_articles: int | None,
) -> tuple[int, list[tuple[str, str]]]:
    new_articles = 0
    failed: list[tuple[str, str]] = []
    with JSONL_PATH.open("a", encoding="utf-8") as sink:
        for window_start, window_end in iter_month_windows(start, end):
            remaining = None if max_articles is None else max_articles - new_articles
            if remaining is not None and remaining <= 0:
                break

            listed = list_month_articles(
                fetcher,
                window_start,
                window_end,
                limit=None if remaining is None else remaining + len(done_links),
                allowed_categories=allowed_categories,
            )
            todo = [(url, category) for url, category in listed.values() if url not in done_links]
            if remaining is not None:
                todo = todo[:remaining]
            print(f"{window_start:%Y-%m}: liet ke {len(listed)} bai, can tai {len(todo)}")

            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = [pool.submit(fetch_article, fetcher, url, category) for url, category in todo]
                for future in as_completed(futures):
                    url, row, error = future.result()
                    if row is None:
                        failed.append((url, error))
                        continue
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    done_links.add(url)
                    new_articles += 1
    return new_articles, failed


def export_all() -> None:
    if not JSONL_PATH.exists():
        print("Chua co bai nao de xuat")
        return
    raw = export_parquet(JSONL_PATH, PARQUET_PATH)
    print(f"Tong trong file: {len(raw)} bai -> {PARQUET_PATH}")
    clean_rows = export_clean_parquet(raw, CLEAN_PARQUET_PATH)
    print(f"Format equity_news_clean_content: {clean_rows} bai -> {CLEAN_PARQUET_PATH}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        export_all()
        return

    fetcher = Fetcher(min_interval=args.delay)
    allowed_categories = None if args.all_categories else load_allowed_categories()
    done_links = load_done_links(JSONL_PATH)
    start = args.start
    while True:
        end = args.end or date.today()
        print(f"Khoang {start} .. {end} | da co {len(done_links)} bai trong {JSONL_PATH.name}")
        # Bi ngat (Ctrl+C) van xuat parquet tu nhung bai da tai; chay lai se tai tiep
        try:
            new_articles, failed = crawl_range(
                fetcher, start, end, done_links, allowed_categories, args.workers, args.max_articles
            )
            print(f"\nBai moi lan nay: {new_articles} | loi/bo qua: {len(failed)}")
            for url, error in failed[:10]:
                print(f"  {error}: {url}")
        finally:
            export_all()

        if args.watch_hours <= 0 or args.end is not None:
            break
        print(f"Cho {args.watch_hours} gio roi crawl lai bai moi...")
        time.sleep(args.watch_hours * 3600)
        start = date.today() - timedelta(days=3)


if __name__ == "__main__":
    main()
