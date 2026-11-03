"""Crawl equity news của stockbiz.vn, cùng cách làm với crawl_vietstock_ver2.py (meta trước, nội dung sau).

LƯU Ý BẢN QUYỀN: robots.txt của stockbiz cho phép crawl (User-agent: * Allow: /) nhưng ghi
"Content-Signal: search=yes, ai-train=no, use=reference" - KHÔNG được dùng nội dung để train hoặc
fine-tune mô hình AI (vd. E1 DAPT, E3 fine-tune PhoBERT). Chỉ dùng để tham chiếu / chấm điểm / thống kê.

stockbiz tổng hợp tin từ báo khác ("Tổng hợp từ Vietnam Finance"): cột source là báo gốc. Site hiện tại
bắt đầu từ 09/2021.

Danh sách bài: GET /<chuyên mục>?trang=<n> (~22 bài/trang, mới -> cũ, không lọc được theo ngày) -
phần duyệt trang dùng chung crawl_store.crawl_paged_meta. Mỗi bài trên danh sách có giờ đăng
(<time datetime>), sapo, báo gốc và mã chứng khoán stockbiz gắn (data-symbol).

Đầu ra: Crawler/data/stockbiz/ (parquet + csv mẫu 1000 bài, xem crawl_store.py).

Ví dụ (từ thư mục gốc dự án):
    python Crawler/crawl_stockbiz.py --phase meta --start 2021-09-01 --end 2026-12-31
    python Crawler/crawl_stockbiz.py --phase content --workers 4
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from urllib.parse import urljoin

from lxml import html as lxml_html

from crawl_store import SiteStore, add_common_args, crawl_paged_meta, run_phases
from crawl_vietstock import OUTPUT_DIR, Fetcher
from crawl_vietstock_ver2 import clean_text

BASE_URL = "https://stockbiz.vn"
DOMAIN = "stockbiz.vn"

# Tỉ lệ is_equity trên bộ thử 2021-2023 (filter_equity_news.py): Doanh nghiệp 92%, Thị trường 96%,
# Tài chính 91%; Kinh tế (nay là "Vĩ mô") 10% nên bỏ. Một bài có thể nằm ở nhiều chuyên mục: khử
# trùng theo link, giữ chuyên mục quét trước. Muốn lấy thêm thì truyền --channels <slug>.
CHANNELS = {
    "doanh-nghiep": "Doanh nghiệp",
    "thi-truong": "Thị trường",
    "tai-chinh": "Tài chính",
}

OUTPUT_COLUMNS = [
    "link", "article_id", "domain", "channel_id", "category", "title", "description", "content",
    "keywords", "author", "source", "symbols", "publication_date", "year",
]
META_COLUMNS = ["link", "article_id", "channel_id", "category", "title", "description", "source", "symbols",
                "publication_date", "month"]
STORE = SiteStore(OUTPUT_DIR / "stockbiz", META_COLUMNS, OUTPUT_COLUMNS)

ARTICLE_ID_PATTERN = re.compile(r"/(\d+)$")
SOURCE_PATTERN = re.compile(r"Tổng hợp từ (.+?)\s*·")
# Bài trên danh sách: article.border-b trong cột chính (lg:col-span-8). Không lấy <aside> (tin nổi bật,
# giống nhau mọi trang) và bài quảng bá thỉnh thoảng được chèn vào cuối danh sách.
LIST_XPATH = ('//div[contains(@class, "lg:col-span-8")]//article[contains(@class, "border-b")]'
              '[not(ancestor::aside)][.//h3/a[contains(@href, "/tin-tuc/")]]')


def iso_to_text(value: str) -> str:
    """"2026-11-03T12:20:00+07:00" -> "2026-11-03 12:20:00" (giờ Việt Nam)."""
    return value.replace("T", " ")[:19]


def symbols_of(node) -> str:
    return ",".join(dict.fromkeys(s.strip() for s in node.xpath(".//*[@data-symbol]/@data-symbol") if s.strip()))


def fetch_page(fetcher: Fetcher, channel: str, page: int) -> list[dict] | None:
    response = fetcher.request("GET", f"{BASE_URL}/{channel}?trang={page}")
    if response is None or response.status_code != 200:
        return None
    document = lxml_html.fromstring(response.content)
    items: list[dict] = []
    for node in document.xpath(LIST_XPATH):
        anchor = node.xpath('.//h3/a[contains(@href, "/tin-tuc/")]')[0]
        times = node.xpath(".//time/@datetime")
        if not times:
            continue
        link = urljoin(BASE_URL, anchor.get("href"))
        texts = [clean_text(p) for p in node.xpath(".//p[not(.//time)]")]
        sources = node.xpath(".//p[.//time]/span")
        article = ARTICLE_ID_PATTERN.search(link)
        items.append({
            "link": link,
            "article_id": article.group(1) if article else "",
            "title": clean_text(anchor),
            "description": texts[0] if texts else "",
            "source": clean_text(sources[0]).rstrip(" ·") if sources else "",
            "symbols": symbols_of(node),
            "publication_date": iso_to_text(times[0]),
        })
    return items


def parse_article(content: bytes, url: str, meta: dict) -> dict | None:
    parser = lxml_html.HTMLParser(encoding="utf-8")
    document = lxml_html.fromstring(content, parser=parser)
    articles = document.xpath('//article[.//h1]')
    if not articles:
        return None
    article = articles[0]
    times = article.xpath(".//time/@datetime")
    if not times:
        return None
    publication_date = iso_to_text(times[0])

    title = clean_text(article.xpath(".//h1")[0])
    sapos = article.xpath('./p[contains(@class, "font-serif")]')
    description = clean_text(sapos[0]) if sapos else meta.get("description", "")
    byline = article.xpath(".//p[.//time]")
    source = SOURCE_PATTERN.search(clean_text(byline[0])) if byline else None

    # Bài chỉ có tóm tắt ("Đọc toàn văn tại <báo gốc>") thì không có article-body -> content rỗng.
    # Một số bài viết thân bài bằng <div> thay vì <p>: lấy các <div> trong cùng.
    bodies = article.xpath('.//div[contains(@class, "article-body")]')
    paragraphs: list[str] = []
    if bodies:
        nodes = bodies[0].xpath(".//p") or bodies[0].xpath(".//div[not(.//div)]")
        paragraphs = [text for text in map(clean_text, nodes) if text]
    if paragraphs and paragraphs[-1] == "Link gốc":
        paragraphs.pop()
    # Dòng cuối ngắn, không có dấu chấm là tên tác giả ("Hải Đường")
    author = ""
    if paragraphs and len(paragraphs[-1].split()) <= 5 and "." not in paragraphs[-1]:
        author = paragraphs.pop()

    return {
        "link": url,
        "article_id": meta.get("article_id", ""),
        "domain": DOMAIN,
        "channel_id": meta.get("channel_id", ""),
        "category": meta.get("category", ""),
        "title": title,
        "description": description,
        "content": "\n".join(p for p in paragraphs if p != title),
        "keywords": "",
        "author": author,
        "source": source.group(1) if source else meta.get("source", ""),
        "symbols": symbols_of(article) or meta.get("symbols", ""),
        "publication_date": publication_date,
        "year": publication_date[:4],
    }


def fetch_article(fetcher: Fetcher, meta: dict) -> tuple[str, dict | None, str, bool]:
    """(link, bài, lỗi, lỗi vĩnh viễn?) - lỗi vĩnh viễn thì lần sau không thử lại."""
    url = meta["link"]
    response = fetcher.request("GET", url)
    if response is None:
        return url, None, "khong tai duoc", False
    if response.status_code in (404, 410):
        return url, None, f"http {response.status_code}", True
    if response.status_code != 200:
        return url, None, f"http {response.status_code}", False
    row = parse_article(response.content, url, meta)
    if row is None:
        return url, None, "khong co thoi gian dang", True
    return url, row, "", False


def crawl_meta(args: argparse.Namespace, fetcher: Fetcher) -> None:
    crawl_paged_meta(args, fetcher, STORE, {c: CHANNELS.get(c, c) for c in args.channels}, fetch_page, last_page_guess=5000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 9, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS), help="slug chuyen muc can quet (mac dinh CHANNELS)")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    run_phases(args, Fetcher(min_interval=args.delay), STORE, crawl_meta, fetch_article)


if __name__ == "__main__":
    main()
