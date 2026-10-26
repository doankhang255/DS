"""Crawl tin vietstock.vn ra đúng 9 cột của dataset_news.parquet (3T):
link, domain, category, title, description, keywords, author, publication_date, year.

Cách lấy danh sách bài: endpoint ajax của chính trang (POST /StartPage/ChannelContentPage
với channelID, page, fromdate, todate; 10 bài/trang). Các kênh cha (chứng khoán, doanh
nghiệp, nhận định - phân tích, kinh tế, tài chính) gộp luôn bài của kênh con, nên chỉ cần
quét 5 kênh cha theo từng tháng. `category` lấy từ nhãn kênh con hiện trên từng dòng của
danh sách (VD "Chính sách", "Cổ phiếu"). Thông tin bài lấy từ thẻ meta của trang bài.

Ví dụ (từ thư mục gốc dự án):
    .venv\\Scripts\\python.exe Crawler\\crawl_vietstock.py --start 2012-03-01 --end 2012-03-31 --max-articles 20
    .venv\\Scripts\\python.exe Crawler\\crawl_vietstock.py            # 2010-01-01 .. hôm nay, chạy tiếp được
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
from datetime import date
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
]

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
JSONL_PATH = OUTPUT_DIR / "vietstock_articles.jsonl"
PARQUET_PATH = OUTPUT_DIR / "vietstock_dataset_news.parquet"

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


def list_month_articles(
    fetcher: Fetcher,
    window_start: date,
    window_end: date,
    limit: int | None,
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


def export_parquet(jsonl_path: Path, parquet_path: Path) -> int:
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
    return len(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--max-articles", type=int, default=None, help="dung sau khi lay du N bai moi (de chay thu)")
    parser.add_argument("--delay", type=float, default=1.0, help="khoang cach toi thieu (giay) giua 2 request")
    parser.add_argument("--workers", type=int, default=1, help="so luong tai bai song song (van chiu --delay chung)")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(min_interval=args.delay)
    done_links = load_done_links(JSONL_PATH)
    print(f"Khoang {args.start} .. {args.end} | da co {len(done_links)} bai trong {JSONL_PATH.name}")

    new_articles = 0
    failed: list[tuple[str, str]] = []
    with JSONL_PATH.open("a", encoding="utf-8") as sink:
        for window_start, window_end in iter_month_windows(args.start, args.end):
            remaining = None if args.max_articles is None else args.max_articles - new_articles
            if remaining is not None and remaining <= 0:
                break

            listed = list_month_articles(
                fetcher,
                window_start,
                window_end,
                limit=None if remaining is None else remaining + len(done_links),
            )
            todo = [(url, category) for url, category in listed.values() if url not in done_links]
            if remaining is not None:
                todo = todo[:remaining]
            print(f"{window_start:%Y-%m}: liet ke {len(listed)} bai, can tai {len(todo)}")

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
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

    total_rows = export_parquet(JSONL_PATH, PARQUET_PATH)
    print(f"\nBai moi lan nay: {new_articles} | loi/bo qua: {len(failed)}")
    for url, error in failed[:10]:
        print(f"  {error}: {url}")
    print(f"Tong trong file: {total_rows} bai -> {PARQUET_PATH}")


if __name__ == "__main__":
    main()
