"""Lọc equity news sau bước gắn công ty: phân loại từng bài theo profile của báo + chuyên mục.

Đọc corpus (equity_news_clean_content.parquet) và firm_symbols.csv (run_firm_extractor.py, có
cột position/mentions), ghi data_news/firm_extract/equity_news_articles.csv, 1 bài 1 dòng:
row, publication_date, domain_norm, category, section, article_type, is_equity, n_symbols,
subject_symbols, symbols, title, link.

Equity news = tin về thị trường cổ phiếu, gồm cả tin của từng công ty lẫn tin thị trường chung;
bước tách tin của từng công ty là subject_symbols (tin thị trường chung không có mã chủ thể).

Chuyên mục chia 2 loại (SECTION_PROFILES):
- "hẹp": mục chỉ đăng tin doanh nghiệp/chứng khoán (vietstock "Cổ tức", "Giao dịch nội bộ",
  baodautu "Sức khỏe doanh nghiệp"...) - bài có mã là equity news.
- "rộng": mục kinh doanh chung, phần lớn bài không về công ty niêm yết nào (diendandoanhnghiep
  "Doanh nghiệp", dantri "Kinh doanh"...) - mã chỉ nhắc qua thì không phải equity news.

article_type:
- "tin tổng hợp nhiều mã": >= MANY_SYMBOLS mã, không mã nào ở tiêu đề/sapo (điểm tin thị
  trường, danh sách giao dịch nội bộ...).
- "tin công ty": có công ty là chủ thể (xem classify) - mã ở tiêu đề/sapo, hoặc được nhắc
  >= SUBJECT_MENTIONS lần trong bài có >= FINANCE_HITS từ ngữ tài chính doanh nghiệp.
- "tin thị trường chung": không có công ty là chủ thể nhưng tiêu đề/sapo có từ ngữ thị trường
  cổ phiếu (MARKET: VN-Index, khối ngoại, cổ phiếu, chứng khoán...).
- "nhắc qua": có mã nhưng chỉ được nhắc qua trong thân bài.
- "không có công ty niêm yết": bài không gắn được mã nào.
- "nội dung rỗng" / "quảng cáo": bài không có công ty chủ thể mà nội dung chỉ là câu dẫn của bảng
  không crawl được (EMPTY_CONTENT, "Top cổ phiếu đáng chú ý...") hoặc tiêu đề quảng cáo khóa học (AD_TITLE).
is_equity: tin công ty / tổng hợp nhiều mã / thị trường chung ở mọi mục; "nhắc qua" chỉ ở mục hẹp.

subject_symbols: công ty chính (mã ở tiêu đề; không có thì ở sapo; không có nữa thì mã được
nhắc nhiều nhất) + công ty phụ được nhắc >= SECONDARY_RATIO số lần của công ty chính. Bài
"Pomina: Cú trượt dài..." (POM 29 lần, HPG 3 lần) chỉ ra POM; bài cả ngành "lợi nhuận ngành
thép" (HPG 16, NKG 16, HSG 12) ra cả 3.

    python Firm_extractor/filter_equity_news.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data_news" / "equity_news_clean_content.parquet"
FIRM_DIR = ROOT / "data_news" / "firm_extract"
SYMBOLS_PATH = FIRM_DIR / "firm_symbols.csv"
OUTPUT_PATH = FIRM_DIR / "equity_news_articles.csv"

# (báo, chuyên mục) -> "rộng"; còn lại là "hẹp". Mục rộng chọn theo tỉ lệ bài gắn được mã
# trên bộ thử (< 40%) và đọc mẫu: bài chủ yếu là quản trị, khởi nghiệp, vĩ mô, chính sách.
SECTION_PROFILES = {
    ("diendandoanhnghiep.vn", "Doanh nghiệp"): "rộng",
    ("dantri.com.vn", "Kinh doanh"): "rộng",
    ("stockbiz.vn", "Kinh tế"): "rộng",
    ("vietstock.vn", "Trái phiếu"): "rộng",
    ("vietstock.vn", "Chính sách"): "rộng",
    ("vietstock.vn", "Phân tích"): "rộng",
}

LEAD_POSITIONS = {"tiêu đề", "sapo"}
SUBJECT_MENTIONS = 2
FINANCE_HITS = 2
MANY_SYMBOLS = 5
SECONDARY_RATIO = 0.5
FINANCE = re.compile(
    r"(?i)lợi nhuận|doanh thu|lãi ròng|lãi trước thuế|lãi sau thuế|thua lỗ|báo lỗ|lỗ ròng|cổ phiếu|"
    r"cổ phần|cổ đông|cổ tức|vốn điều lệ|tăng vốn|phát hành|niêm yết|thoái vốn|sáp nhập|m&a|trái phiếu|"
    r"đhđcđ|đại hội đồng cổ đông|hđqt|hội đồng quản trị|kết quả kinh doanh|báo cáo tài chính|kiểm toán|"
    r"ipo|vốn hóa|thị giá|chuyển nhượng|nợ vay|dòng tiền|kế hoạch kinh doanh"
)
# Từ ngữ thị trường cổ phiếu ở tiêu đề/sapo. Không dùng từ quá chung: "chỉ số" (chỉ số PAPI),
# "thanh khoản" (ngân hàng), "nhà đầu tư nước ngoài" (FDI), "ký quỹ" (ký quỹ lữ hành).
MARKET = re.compile(
    r"(?i)vn-?index|hnx-?index|upcom-?index|\bvn30\b|\bvn100\b|\bttck\b|chứng khoán|cổ phiếu|khối ngoại|"
    r"\betfs?\b|bán ròng|mua ròng|tự doanh|\bmargin\b|phiên giao dịch|vốn hóa thị trường|niêm yết"
)
# Bài "Top cổ phiếu đáng chú ý...": nội dung chỉ có câu dẫn, bảng mã không crawl được
EMPTY_CONTENT = re.compile(r"(?i)danh sách các mã (?:cổ phiếu )?tăng")
EMPTY_CONTENT_CHARS = 300
# Quảng cáo khóa học (chủ yếu Vietstock tự quảng cáo)
AD_TITLE = re.compile(r"(?i)khóa học|khoá học|khai giảng|ưu đãi học phí|lớp học")


def classify(symbols: pd.DataFrame, finance_hits: int, market: bool) -> tuple[str, list[str]]:
    """(article_type, các mã là chủ thể) của 1 bài từ các dòng firm_symbols.csv của bài đó.

    market: tiêu đề/sapo có từ ngữ thị trường cổ phiếu (MARKET).
    """
    no_subject = "tin thị trường chung" if market else "nhắc qua"
    if symbols.empty:
        return ("tin thị trường chung" if market else "không có công ty niêm yết"), []
    lead = symbols["position"].isin(LEAD_POSITIONS)
    if not lead.any() and len(symbols) >= MANY_SYMBOLS:
        return "tin tổng hợp nhiều mã", []
    mentions = symbols["mentions"]
    if (symbols["position"] == "tiêu đề").any():
        main = symbols["position"] == "tiêu đề"
    elif lead.any():
        main = lead
    elif mentions.max() >= SUBJECT_MENTIONS and finance_hits >= FINANCE_HITS:
        main = mentions == mentions.max()
    else:
        return no_subject, []
    # Công ty phụ: nhắc gần bằng công ty chính (và không chỉ 1 lần), không phải nhắc qua
    top = mentions[main].max()
    secondary = ~main & (mentions >= SUBJECT_MENTIONS) & (mentions >= SECONDARY_RATIO * top)
    return "tin công ty", sorted(symbols.loc[main | secondary, "symbol"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = pd.read_parquet(CORPUS_PATH, columns=["publication_date", "domain_norm", "category", "title",
                                                   "description", "content", "link"]).reset_index(drop=True)
    pairs = pd.read_csv(SYMBOLS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    pairs["row"] = pairs["row"].astype(int)
    pairs["mentions"] = pairs["mentions"].astype(int)
    by_row = dict(tuple(pairs.groupby("row")))
    empty = pairs.iloc[0:0]

    text = corpus["title"].fillna("") + "\n" + corpus["description"].fillna("") + "\n" + corpus["content"].fillna("")
    finance_hits = text.str.count(FINANCE.pattern)
    market = (corpus["title"].fillna("") + "\n" + corpus["description"].fillna("")).str.contains(MARKET)
    content = corpus["content"].fillna("")
    empty_content = content.str.contains(EMPTY_CONTENT) & (content.str.len() < EMPTY_CONTENT_CHARS)
    advert = corpus["title"].fillna("").str.contains(AD_TITLE)

    records = []
    for row, article in corpus.iterrows():
        symbols = by_row.get(row, empty)
        article_type, subject = classify(symbols, int(finance_hits[row]), bool(market[row]))
        if article_type != "tin công ty":
            if empty_content[row]:
                article_type = "nội dung rỗng"
            elif advert[row]:
                article_type = "quảng cáo"
        section = SECTION_PROFILES.get((article["domain_norm"], article["category"]), "hẹp")
        is_equity = (article_type in {"tin công ty", "tin tổng hợp nhiều mã", "tin thị trường chung"}
                     or (section == "hẹp" and article_type == "nhắc qua"))
        records.append({
            "row": row,
            "publication_date": article["publication_date"],
            "domain_norm": article["domain_norm"],
            "category": article["category"],
            "section": section,
            "article_type": article_type,
            "is_equity": is_equity,
            "n_symbols": len(symbols),
            "subject_symbols": ",".join(subject),
            "symbols": ",".join(symbols["symbol"]),
            "title": article["title"],
            "link": article["link"],
        })
    out = pd.DataFrame(records)
    tmp = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, OUTPUT_PATH)

    print(f"{OUTPUT_PATH.name}: {out['is_equity'].sum():,}/{len(out):,} bài là equity news "
          f"({out['is_equity'].mean():.1%})")
    print(out["article_type"].value_counts().to_string())
    summary = out.groupby("domain_norm").agg(bai=("row", "size"), equity=("is_equity", "mean"))
    summary["equity"] = (summary["equity"] * 100).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
