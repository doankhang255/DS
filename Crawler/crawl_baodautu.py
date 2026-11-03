"""Crawl equity news của baodautu.vn, cùng cách làm với crawl_vietstock_ver2.py (meta trước, nội dung sau).

Danh sách bài: GET /<kênh>/p<trang> (khoảng 12 bài/trang, mới -> cũ, không lọc được theo ngày) -
phần duyệt trang dùng chung crawl_store.crawl_paged_meta. Trang danh sách KHÔNG ghi ngày đăng:
phase meta lấy ngày gần đúng từ đường dẫn ảnh thumbnail (/Images/<người đăng>/YYYY/MM/DD/...,
ngày tải ảnh lên, thường là ngày đăng); phase content ghi đè bằng giờ đăng thật trên trang bài.

- --phase meta: link, article_id (số sau "-d" trong link), channel_id (slug kênh), category, title,
  description (sapo), publication_date (ngày gần đúng, xem trên).
- --phase content: tải trang bài: tiêu đề, sapo, content, keywords (tag), author, giờ đăng.
- Mặc định (--phase all) chạy meta rồi content. --shard / --phase export như crawl_vietstock_ver2.py.

Đầu ra: Crawler/data/baodautu/ (parquet + csv mẫu 1000 bài, xem crawl_store.py).

Ví dụ (từ thư mục gốc dự án):
    python Crawler/crawl_baodautu.py --phase meta --start 2019-01-01 --end 2026-12-31
    python Crawler/crawl_baodautu.py --phase content --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date

from lxml import html as lxml_html

from crawl_store import SiteStore, add_common_args, crawl_paged_meta, run_phases
from crawl_vietstock import OUTPUT_DIR, Fetcher, first_attribute
from crawl_vietstock_ver2 import clean_text

BASE_URL = "https://baodautu.vn"
DOMAIN = "baodautu.vn"

# Bộ thử chỉ có "Sức khỏe doanh nghiệp" (d53): 99% equity (filter_equity_news.py), 95% bài có mã
# công ty ở tiêu đề/sapo; mục này có từ 10/2019 và ngừng cập nhật từ 01/2026. "Đầu tư tài chính"
# (d6 - tài chính, chứng khoán, ngân hàng) vẫn cập nhật: ~45% bài có mã ở tiêu đề/sapo (mẫu 2022-2026),
# phần còn lại phần lớn là tin thị trường chứng khoán / ngân hàng. Các mục khác (Thời sự, Kinh doanh,
# Địa ốc, Tiêu dùng...) là tin rộng, không lấy. Muốn lấy thêm thì truyền --channels <slug>.
CHANNELS = {
    "suc-khoe-doanh-nghiep-d53": "Sức khỏe doanh nghiệp",
    "dau-tu-tai-chinh-d6": "Đầu tư tài chính",
}

OUTPUT_COLUMNS = [
    "link", "article_id", "domain", "channel_id", "category", "title", "description", "content",
    "keywords", "author", "source", "publication_date", "year",
]
META_COLUMNS = ["link", "article_id", "channel_id", "category", "title", "description", "publication_date", "month"]
STORE = SiteStore(OUTPUT_DIR / "baodautu", META_COLUMNS, OUTPUT_COLUMNS)

ARTICLE_ID_PATTERN = re.compile(r"-d(\d+)\.html$")
IMAGE_DATE_PATTERN = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")
POST_TIME_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})")
# Trong thân bài: <table> là ảnh + chú thích, .style-box là hộp giới thiệu bài khác
CONTENT_XPATH = './/p[not(ancestor::table) and not(ancestor::*[contains(@class, "style-box")])]'


def fetch_page(fetcher: Fetcher, channel: str, page: int) -> list[dict] | None:
    response = fetcher.request("GET", f"{BASE_URL}/{channel}/p{page}")
    if response is None or response.status_code != 200:
        return None
    document = lxml_html.fromstring(response.content)
    items: list[dict] = []
    for node in document.xpath('//ul[contains(@class, "list_news_home")]/li'):
        anchors = [a for a in node.xpath('.//a[contains(@href, ".html")]') if a.text_content().strip()]
        if not anchors:
            continue
        link = anchors[0].get("href")
        images = node.xpath(".//img/@src")
        match = IMAGE_DATE_PATTERN.search(images[0]) if images else None
        sapos = node.xpath('.//*[contains(@class, "sapo")]')
        article = ARTICLE_ID_PATTERN.search(link)
        items.append({
            "link": link if link.startswith("http") else BASE_URL + link,
            "article_id": article.group(1) if article else "",
            "title": clean_text(anchors[0]),
            "description": clean_text(sapos[0]) if sapos else "",
            "publication_date": "-".join(match.groups()) if match else "",
        })
    # Bài không có ảnh: lấy ngày của bài liền sau/trước (danh sách xếp mới -> cũ); cả trang không
    # có ảnh thì bỏ bài đó (ngày rỗng sẽ bị coi là < --start và dừng duyệt sớm)
    for index, item in enumerate(items):
        if not item["publication_date"]:
            neighbours = [other["publication_date"] for other in items[index + 1:] + items[:index][::-1] if other["publication_date"]]
            item["publication_date"] = neighbours[0] if neighbours else ""
    return [item for item in items if item["publication_date"]]


def parse_article(content: bytes, url: str, meta: dict) -> dict | None:
    parser = lxml_html.HTMLParser(encoding="utf-8")
    document = lxml_html.fromstring(content, parser=parser)

    times = document.xpath('//span[contains(@class, "post-time")]')
    match = POST_TIME_PATTERN.search(clean_text(times[0])) if times else None
    if not match:
        return None
    day, month, year, hour, minute = match.groups()
    publication_date = f"{year}-{month}-{day} {hour}:{minute}:00"

    titles = document.xpath('//div[contains(@class, "title-detail")]')
    title = clean_text(titles[0]) if titles else first_attribute(document, '//meta[@property="og:title"]/@content')
    sapos = document.xpath('//div[contains(@class, "sapo_detail")]')
    description = clean_text(sapos[0]) if sapos else meta.get("description", "")

    paragraphs: list[str] = []
    bodies = document.xpath('//*[@id="content_detail_news"]')
    if bodies:
        paragraphs = [text for text in map(clean_text, bodies[0].xpath(CONTENT_XPATH)) if text and text != title]

    tags = [clean_text(a).lstrip("#").strip() for a in document.xpath('//a[contains(@class, "tag_detail_item")]')]
    authors = document.xpath('//a[contains(@class, "author")]')

    return {
        "link": url,
        "article_id": meta.get("article_id", ""),
        "domain": DOMAIN,
        "channel_id": meta.get("channel_id", ""),
        "category": meta.get("category", ""),
        "title": title,
        "description": description,
        "content": "\n".join(paragraphs),
        "keywords": json.dumps([t for t in tags if t], ensure_ascii=False) if tags else "",
        "author": clean_text(authors[0]) if authors else "",
        "source": "",  # bài của chính baodautu
        "publication_date": publication_date,
        "year": year,
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
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS), help="slug kenh can quet (mac dinh CHANNELS)")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    run_phases(args, Fetcher(min_interval=args.delay), STORE, crawl_meta, fetch_article)


if __name__ == "__main__":
    main()
