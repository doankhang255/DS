"""Crawl equity news của vietstock.vn, có NỘI DUNG bài (ver1 chỉ lấy thẻ meta).

Khác ver1:
- Chỉ quét các kênh con là tin cổ phiếu / doanh nghiệp (EQUITY_CHANNELS), không quét cả 5 kênh
  cha (kinh tế, tài chính... phần lớn không phải equity news).
- Lấy nội dung bài từ khung #vst_detail của trang bài: pHead = sapo (description), pSubTitle +
  pBody = content (mỗi đoạn 1 dòng, như cột content của equity_news_clean_content.parquet).
  Bảng trong bài hầu hết là ảnh biểu đồ / file đính kèm nên không lấy.
- Chạy 2 phase, mỗi phase chạy tiếp được khi bị ngắt:
  1. --phase meta: lấy các cột có sẵn trên trang danh sách (POST /StartPage/ChannelContentPage
     như ver1, 10 bài/trang, nhận cả channelID kênh con): link, article_id, channel_id, category,
     title, publication_date (đến phút), description (sapo bị cắt ngắn "..."). Chỉ ~1 request /
     10 bài nên nhanh. (kênh, tháng) đã qua và liệt kê hết thì ghi vào windows.parquet, lần sau
     không liệt kê lại; tháng hiện tại lần nào cũng liệt kê lại để lấy bài mới.
  2. --phase content: tải trang từng bài trong meta: sapo đầy đủ, content, keywords, author,
     source, thời gian đăng đến giây (đều nằm trên trang bài nên lấy chung 1 request). Bỏ qua bài
     đã có. Bài lỗi vĩnh viễn (404, không có thời gian đăng) ghi vào failed.parquet và bỏ qua; lỗi
     mạng thì lần sau thử lại. --shard K/N chia meta cho N máy (máy thứ K lấy bài có
     article_id % N == K - 1); chép article_parts/ của các máy về 1 máy rồi chạy --phase export.
  Mặc định (--phase all) chạy meta rồi content.

Đầu ra: Crawler/data/vietstock/ (parquet, xem crawl_store.py) - meta.parquet, news.parquet và
meta_sample_1000.csv, news_sample_1000.csv (1000 bài ngẫu nhiên để kiểm tra). Cột của news:
link, article_id, domain, channel_id, category, title, description, content, keywords, author,
source, publication_date, year. Dữ liệu .jsonl của bản cũ được tự chuyển sang parquet ở lần chạy
đầu (file cũ cất vào Crawler/data/legacy/).

Ví dụ (từ thư mục gốc dự án):
    python Crawler/crawl_vietstock_ver2.py --phase meta --start 2010-01-01 --end 2015-12-31 --workers 4 --delay 0.5
    python Crawler/crawl_vietstock_ver2.py --phase content --start 2010-01-01 --end 2010-12-31 --workers 4 --delay 0.5
    python Crawler/crawl_vietstock_ver2.py --phase content --shard 1/3    # máy 1 trong 3 máy
    python Crawler/crawl_vietstock_ver2.py --start 2024-01-01 --end 2024-01-31 --max-articles 20
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import urljoin

from lxml import html as lxml_html

from crawl_vietstock import (
    BASE_URL,
    DOMAIN,
    LIST_ENDPOINT,
    OUTPUT_DIR,
    PAGE_SIZE,
    TIMEZONE_SUFFIX_PATTERN,
    TITLE_SUFFIX_PATTERN,
    TOTAL_ROWS_PATTERN,
    Fetcher,
    first_attribute,
    iter_month_windows,
)
from crawl_store import SiteStore, add_common_args, run_phases

# Kênh con của "Chứng khoán" (144), "Doanh nghiệp" (733), "Nhận định - Phân tích" (579), channelID
# lấy từ trang của từng kênh (biến channelID trong HTML). Chọn theo tỉ lệ is_equity trên bộ thử
# 2010-2023 (filter_equity_news.py): giữ các kênh >= 60%. Nhân vật (214) không có trong bộ thử;
# đo trên mẫu 06/2012, 06/2015, 06/2024 (30 bài, classify của filter_equity_news.py): 80% là tin
# công ty (bổ nhiệm, HĐQT của công ty niêm yết) nên giữ. Bỏ (cùng mẫu đo): Trái phiếu (785, 57%
# trên bộ thử - trái phiếu chính phủ, lãi suất), Phái sinh (4186), Ý kiến chuyên gia (145 - nhận
# định của công ty chứng khoán), Câu chuyện đầu tư (3355), Nhận định thị trường (1636) / Phân tích
# kỹ thuật (585) (tin thị trường chung), Ngân hàng (757, 27% - phần lớn quảng bá sản phẩm),
# Kinh tế - Đầu tư (768, 27%), Bảo hiểm, Bất động sản, Vĩ mô, Thuế, Hàng hóa (<= 22%).
# Muốn lấy thêm thì truyền --channels.
EQUITY_CHANNELS = {
    737: "Hoạt động kinh doanh",  # 98% equity trên bộ thử
    739: "Giao dịch nội bộ",  # 99%
    830: "Cổ phiếu",  # 83%
    738: "Cổ tức",  # 99%
    764: "Tăng vốn - M&A",  # 96%
    741: "Niêm yết",  # 100%
    746: "IPO - Cổ phần hóa",  # 64%
    3358: "ETF và các quỹ",  # 95%
    3118: "Trái phiếu doanh nghiệp",  # 81%
    143: "Chính sách",  # 82%
    582: "Phân tích cơ bản",  # 98%
    4308: "Chứng quyền",  # 92%
    214: "Nhân vật",  # 80% tin công ty trên mẫu 30 bài
}

OUTPUT_COLUMNS = [
    "link",
    "article_id",
    "domain",
    "channel_id",
    "category",
    "title",
    "description",
    "content",
    "keywords",
    "author",
    "source",
    "publication_date",
    "year",
]

META_COLUMNS = ["link", "article_id", "channel_id", "category", "title", "description", "publication_date", "month"]
STORE = SiteStore(OUTPUT_DIR / "vietstock", META_COLUMNS, OUTPUT_COLUMNS)
LEGACY_PREFIX = "vietstock_equity"  # file .jsonl của bản trước: Crawler/data/vietstock_equity_*.jsonl

# Link bài: /2026/11/<slug>-<channelID>-<articleID>.htm
ARTICLE_URL_PATTERN = re.compile(r"-(\d+)-(\d+)\.htm$")
# Thời gian trên trang danh sách: "30/03 18:56" (không có năm); bài mới đăng thì là "2 giờ trước"
LIST_TIME_PATTERN = re.compile(r"^(\d{2})/(\d{2}) (\d{2}:\d{2})$")
# Đoạn của nội dung bài, theo thứ tự trong bài; class có thể kèm class khác ("pBody right").
# Tin thông báo (niêm yết, cổ tức...) hay viết thân bài thành danh sách <li> thay vì pBody;
# <li> trong bảng là chú thích ảnh / file đính kèm nên bỏ.
CONTENT_XPATH = (
    './/p[contains(concat(" ", normalize-space(@class), " "), " pBody ")'
    ' or contains(concat(" ", normalize-space(@class), " "), " pSubTitle ")]'
    " | .//li[not(ancestor::table)]"
)
# Đoạn không phải nội dung, dùng khi bài không có pBody (lấy mọi <p> còn lại của #vst_detail)
NON_CONTENT_CLASSES = {"pTitle", "pHead", "pAuthor", "pSource", "pPublishTimeSource", "pCaption"}


def article_id(url: str) -> str:
    match = ARTICLE_URL_PATTERN.search(url)
    return match.group(2) if match else ""


def clean_text(node) -> str:
    return " ".join(node.text_content().split())


def first_text(node, class_name: str) -> str:
    """Chữ của <p> đầu tiên có class `class_name` trong node ("" nếu không có)."""
    if node is None:
        return ""
    found = node.xpath(f'.//p[contains(concat(" ", normalize-space(@class), " "), " {class_name} ")]')
    return clean_text(found[0]) if found else ""


def parse_article(content: bytes, url: str, list_category: str, channel_id: int) -> dict | None:
    parser = lxml_html.HTMLParser(encoding="utf-8")
    document = lxml_html.fromstring(content, parser=parser)

    published_raw = first_attribute(document, '//meta[@property="article:published_time"]/@content')
    if not published_raw:
        published_raw = first_attribute(document, '//meta[@name="pubdate"]/@content')
    if not published_raw:
        return None
    publication_date = TIMEZONE_SUFFIX_PATTERN.sub("", published_raw).replace("T", " ")

    details = document.xpath('//div[@id="vst_detail"]')
    detail = details[0] if details else None

    title = first_text(detail, "pTitle") or TITLE_SUFFIX_PATTERN.sub("", first_attribute(document, "//title/text()"))

    description = first_text(detail, "pHead")
    if not description:
        description = first_attribute(document, '//meta[@name="description"]/@content')
    if description == title:
        description = ""

    paragraphs: list[str] = []
    if detail is not None:
        nodes = detail.xpath(CONTENT_XPATH)
        if not nodes:
            nodes = [p for p in detail.xpath(".//p") if not NON_CONTENT_CLASSES & set((p.get("class") or "").split())]
        # Bài cũ có thêm 1 <p> không class ở cuối lặp lại tiêu đề
        paragraphs = [text for text in map(clean_text, nodes) if text and text != title]

    tags = {}
    for anchor in document.xpath('//ul[li[contains(@class, "tag_list")]]/li/a[starts-with(@href, "/tag/")]'):
        tag_name = anchor.text_content().strip()
        if tag_name:
            tags[tag_name] = anchor.get("href")

    return {
        "link": url,
        "article_id": article_id(url),
        "domain": DOMAIN,
        "channel_id": str(channel_id),
        "category": list_category or EQUITY_CHANNELS.get(channel_id, ""),
        "title": title,
        "description": description,
        "content": "\n".join(paragraphs),
        "keywords": json.dumps(tags, ensure_ascii=False) if tags else "",
        "author": first_text(detail, "pAuthor"),
        "source": first_text(detail, "pSource"),
        "publication_date": publication_date,
        "year": publication_date[:4],
    }


def parse_list_items(fragment: str, year: int) -> tuple[list[dict], int]:
    """(các bài trên 1 trang danh sách, tổng số bài của khoảng ngày). `year`: năm của khoảng ngày
    (khoảng ngày luôn nằm trong 1 tháng), vì thời gian trên danh sách không ghi năm."""
    total_match = TOTAL_ROWS_PATTERN.search(fragment)
    total_rows = int(total_match.group(1)) if total_match else 0
    if not fragment.strip():
        return [], total_rows

    document = lxml_html.fromstring(f"<div>{fragment}</div>")
    items: list[dict] = []
    for node in document.xpath('//div[contains(@class, "channelContent")]'):
        hrefs = node.xpath('.//div[contains(@class, "post_img")]//a/@href')
        if not hrefs:
            continue
        meta = node.xpath('.//div[contains(@class, "meta3")]/a')
        time_match = LIST_TIME_PATTERN.match(meta[1].get("title", "").strip()) if len(meta) > 1 else None
        titles = node.xpath(".//h4/a")
        summaries = node.xpath('.//p[contains(@class, "post-p")]')
        items.append({
            "link": urljoin(BASE_URL, hrefs[0]),
            "category": meta[0].get("title", "").strip() if meta else "",
            # Tiêu đề, sapo trên danh sách bị escape 2 lần ("&amp;nbsp;"); sapo còn bị cắt ngắn
            "title": " ".join(html.unescape(titles[0].text_content()).split()) if titles else "",
            "description": " ".join(html.unescape(summaries[0].text_content()).split()) if summaries else "",
            "publication_date": f"{year}-{time_match[2]}-{time_match[1]} {time_match[3]}:00" if time_match else "",
        })
    return items, total_rows


def list_channel_window(
    fetcher: Fetcher,
    channel_id: int,
    window_start: date,
    window_end: date,
) -> tuple[list[dict], bool]:
    """(các bài của 1 kênh trong 1 khoảng ngày, đã liệt kê hết chưa)."""
    found: list[dict] = []
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
            print(f"  [bo qua] kenh {channel_id} trang {page}: khong lay duoc danh sach")
            return found, False
        items, total_rows = parse_list_items(response.text, window_start.year)
        found.extend(items)
        if not items or page * PAGE_SIZE >= total_rows:
            return found, True
        page += 1


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
    row = parse_article(response.content, url, meta["category"], int(meta["channel_id"]))
    if row is None:
        return url, None, "khong co thoi gian dang", True
    return url, row, "", False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--channels", type=int, nargs="+", default=list(EQUITY_CHANNELS), help="channelID can quet (mac dinh EQUITY_CHANNELS)")
    add_common_args(parser)
    return parser.parse_args()


def crawl_meta(args: argparse.Namespace, fetcher: Fetcher) -> None:
    known = STORE.known_meta_links()
    done_windows = STORE.done_windows()
    print(f"[meta] {args.start} .. {args.end} | {len(args.channels)} kenh | da co {len(known)} bai, "
          f"{len(done_windows)} (kenh, thang) xong")

    writer = STORE.meta_writer()
    new_links = 0
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    for window_start, window_end in iter_month_windows(args.start, args.end):
        month = f"{window_start:%Y-%m}"
        # Chỉ đánh dấu xong khi quét trọn 1 tháng đã qua (tháng hiện tại còn bài mới;
        # --start/--end giữa tháng thì chưa đủ tháng)
        last_day = calendar.monthrange(window_start.year, window_start.month)[1]
        full_past_month = window_start.day == 1 and window_end.day == last_day and window_end < date.today()
        month_links = 0
        finished: list[dict] = []
        # Các kênh liệt kê song song (vẫn chịu --delay chung); ghi file ở luồng chính
        futures = {
            pool.submit(list_channel_window, fetcher, channel_id, window_start, window_end): channel_id
            for channel_id in args.channels
            if (str(channel_id), month) not in done_windows
        }
        for future in as_completed(futures):
            channel_id = futures[future]
            items, complete = future.result()
            for item in items:
                if item["link"] in known:
                    continue
                writer.add({
                    "link": item["link"],
                    "article_id": article_id(item["link"]),
                    "channel_id": channel_id,
                    "category": item["category"] or EQUITY_CHANNELS.get(channel_id, ""),
                    "title": item["title"],
                    "description": item["description"],
                    "publication_date": item["publication_date"],
                    "month": month,
                })
                known.add(item["link"])
                month_links += 1
            if complete and full_past_month:
                finished.append({"channel_id": channel_id, "month": month, "articles": len(items)})
        # Ghi meta của tháng ra đĩa rồi mới đánh dấu (kênh, tháng) xong
        writer.flush()
        STORE.mark_windows(finished)
        done_windows.update((str(row["channel_id"]), month) for row in finished)
        new_links += month_links
        print(f"{month}: +{month_links} bai (tong {len(known)})")
    pool.shutdown()
    print(f"[meta] bai moi lan nay: {new_links}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    STORE.migrate_legacy(LEGACY_PREFIX)
    run_phases(args, Fetcher(min_interval=args.delay), STORE, crawl_meta, fetch_article)


if __name__ == "__main__":
    main()
