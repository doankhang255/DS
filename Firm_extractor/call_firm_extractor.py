"""Tách mã chứng khoán cho từng bài tin (offline, theo vn_stock_symbols.csv).

Cấu trúc file:
  PHẦN 0 - Danh sách mã và tên công ty (đọc vn_stock_symbols.csv, nhóm cùng công ty)
  PHẦN 1 - Luật cho MÃ chứng khoán viết trong bài       -> find_symbols_in_text
  PHẦN 2 - Luật cho TÊN công ty viết trong bài           -> find_names_in_text
  PHẦN 3 - Kết hợp mã và tên thành kết quả của 1 bài     -> extract_symbols_offline
"""
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ===========================================================================
# PHẦN 0 - DANH SÁCH MÃ VÀ TÊN CÔNG TY
# ===========================================================================

SYMBOLS_PATH = Path(__file__).resolve().parent.parent / "data_news" / "vn_stock_symbols.csv"

# Cùng 1 công ty mang nhiều mã (đổi mã khi chuyển sàn): tin về công ty được gán cho tất cả
# các mã trong nhóm, và tên nào trong nhóm cũng khớp ra cả nhóm. vn_stock_symbols.csv mỗi
# mã chỉ có 1 tên (tên hiện tại), nên tên cũ của nhóm khai báo ở đây.
SAME_COMPANY_SYMBOLS = [
    (("ANI", "SIC"), ("CTCP Đầu tư - Phát triển Sông Đà", "Công ty Cổ phần ANI")),
]
# Mã trong nhóm luôn xếp theo thứ tự chữ cái, để so được với tuple mã của tên dùng chung
# (load_symbols cũng xếp) dù khai báo ở trên theo thứ tự nào
SAME_COMPANY_GROUPS = {tuple(sorted(symbols)) for symbols, _ in SAME_COMPANY_SYMBOLS}
COMPANY_GROUP_OF = {symbol: group for group in SAME_COMPANY_GROUPS for symbol in group}


def read_symbol_names() -> pd.DataFrame:
    """vn_stock_symbols.csv, mỗi tên 1 dòng: tên chính thức (organ_name) và tên cũ trước khi đổi
    tên (old_names, tìm từ kho tin - xem review_old_names.csv); mỗi cột có thể nhiều tên, cách
    nhau " | "."""
    raw = pd.read_csv(SYMBOLS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    names = raw["organ_name"] + "|" + raw["old_names"]
    sym = raw.assign(organ_name=names.str.split("|")).explode("organ_name")
    sym["organ_name"] = sym["organ_name"].str.strip()
    # Mã rời sàn không có tên vẫn giữ 1 dòng tên rỗng (bị bỏ ở load_symbols); bỏ ô old_names trống
    return sym[(sym["organ_name"] != "") | ~sym.duplicated("symbol")]


@lru_cache(maxsize=1)
def load_symbols() -> tuple[frozenset, dict, dict, dict]:
    """(tập mã cổ phiếu, tên chuẩn hoá -> mã, lõi tên -> mã, tên phụ -> mã); mỗi tên -> tuple mã.

    Tên chuẩn hoá và lõi tên theo luật ở PHẦN 2 (normalize_name, name_core).
    Tên phụ: lõi bỏ chữ "tổng" ("Tổng công ty IDICO" -> "idico"), chỉ dùng khi không trùng
    lõi của công ty khác. Không bỏ chữ "ngân hàng": ngân hàng không được viết là "CTCP/Cty/
    Tập đoàn" + tên ("Cty Hàng hải" không phải Ngân hàng TMCP Hàng Hải - MSB). Nhiều công ty khác nhau có đúng cùng tên chính thức
    ("CTCP Nam Việt" = ANV và NAV, "CTCP Xây dựng Số 5" = SC5 và VC5) thì tên đó giữ đủ các
    mã - tên đó không tự ra mã được, xem extract_symbols_offline; lõi/tên phụ trùng giữa các
    công ty tên khác nhau thì bỏ, không đoán. Nhóm trong SAME_COMPANY_SYMBOLS: mọi tên của
    nhóm khớp ra cả nhóm.
    """
    sym = read_symbol_names()
    full = sym["organ_name"].map(normalize_name)  # mã rời sàn không có tên -> "", bị bỏ ở build_index
    core = full.map(name_core)
    alias = core.where(core.str.startswith("tổng ")).str[len("tổng "):]

    def build_index(keys: pd.Series) -> dict[str, tuple[str, ...]]:
        """Tên -> các mã. Giữ khi tên chỉ thuộc 1 mã (kể cả nhiều tên của cùng mã trùng lõi),
        hoặc nhiều mã có đúng cùng tên chính thức; trùng lõi giữa các tên khác nhau của
        nhiều mã thì bỏ."""
        groups: dict[str, list[tuple[str, str]]] = {}
        for key, symbol, full_name in zip(keys, sym["symbol"], full):
            if isinstance(key, str) and len(key) >= 4:
                groups.setdefault(key, []).append((symbol, full_name))
        index = {}
        for key, items in groups.items():
            symbols = tuple(sorted({s for s, _ in items}))
            if len(symbols) == 1 or len({f for _, f in items}) == 1:
                index[key] = symbols
        return index

    by_full = build_index(full)
    by_core = build_index(core)
    core_keys = set(core)
    by_alias = {k: s for k, s in build_index(alias).items() if k not in core_keys}
    for symbols, names in SAME_COMPANY_SYMBOLS:
        group = tuple(sorted(symbols))
        for name in names:
            key = normalize_name(name)
            by_full[key] = group
            if len(name_core(key)) >= 4:
                by_core[name_core(key)] = group
    return frozenset(sym["symbol"]), by_full, by_core, by_alias


@lru_cache(maxsize=1)
def securities_symbols() -> frozenset:
    """Mã của các công ty chứng khoán (tên trong danh sách có chữ "chứng khoán")."""
    return sector_symbols("chứng khoán")


@lru_cache(maxsize=None)
def sector_symbols(sector: str) -> frozenset:
    """Mã của các công ty có tên chứa cụm chữ `sector` ("chứng khoán", "bảo hiểm")."""
    sym = read_symbol_names()
    return frozenset(sym.loc[sym["organ_name"].map(normalize_name).str.contains(rf"\b{sector}\b"), "symbol"])


# ===========================================================================
# PHẦN 1 - LUẬT CHO MÃ CHỨNG KHOÁN VIẾT TRONG BÀI
# ===========================================================================

# Luật 1.1 - Mã dễ nhầm: mã đứng trơn chỉ được nhận khi bài có thêm bằng chứng - mã đó kèm tên
# sàn hoặc ngay sau chữ "Chứng khoán" (luật 1.3), hoặc có tên công ty của mã (PHẦN 2).
# Đưa vào danh sách khi mã viết hoa trùng với:
# - từ viết tắt / thuật ngữ thông dụng: tiền tệ, chức danh, loại hình doanh nghiệp, chỉ số
#   kinh tế, xếp hạng...;
# - viết tắt cơ quan nhà nước: bộ, ủy ban, ngân hàng nhà nước...;
# - tên, thương hiệu của công ty / quỹ / sản phẩm khác, hoặc tên cũ của công ty khác.
# Tiêu chí trên toàn kho tin: mã đứng trơn >= 25 lần nhưng có tên sàn đi kèm < 5% số lần;
# mã có tỉ lệ cao hơn chỉ đưa vào khi đã xem ngữ cảnh và phần đứng trơn chủ yếu là nghĩa khác.
# Không đưa vào mã đứng trơn vẫn đúng là công ty đó (kể cả khi công ty chưa niêm yết trong
# giai đoạn của kho tin).
# TODO (03/11/2026, kiểm tra kênh vietstock): còn nhận nhầm 2 mã, cần xem ngữ cảnh rồi thêm vào đây:
# - SMA: bài phân tích kỹ thuật, "SMA" là đường trung bình động (simple moving average).
# - LAI: bài "Chứng khoán phái sinh ngày ...".
AMBIGUOUS_SYMBOLS = {
    "AVG", "BCA", "BCR", "BIG", "BOT", "BSC", "BSL", "BWA", "CAR", "CCP", "CDH", "CEO",
    "CGV", "CHC", "CIP", "CMC", "CNC", "CNN", "CPH", "CPI", "CSI", "DAS", "DCF", "DCG",
    "DIC", "DMS", "DNA", "DNT", "DOC", "DWS", "EPC", "FCC", "FSO", "GIC", "HAC", "HDS",
    "HSV", "ICT", "MDF", "MEF", "MTA", "MTV", "NAV", "NDT", "NJC", "NO1", "PLE", "QNP",
    "SBG", "SCG", "SGI", "SPC", "SPV", "THM", "TIN", "TPS", "TTC", "USD", "VDB", "VEC",
    "VFS", "VNX", "VPA", "VPS", "VRG", "VTM", "ALC", "ANI", "AUM", "BQP", "HPA", "SCA",
    "TCX", "TEC", "UPS", "BTC", "SBV", "SSC", "TCT", "TOP", "IFC",
}

# Mã dễ nhầm theo bài: mã trùng mã/viết tắt của một tổ chức khác, nhưng ngoài các bài nói về tổ
# chức đó thì mã đứng trơn vẫn đúng là công ty. Bài có cụm chữ khớp mẫu thì mã đứng trơn cũng cần
# bằng chứng như mã dễ nhầm (luật 1.1). Quỹ ETF / quỹ đầu tư niêm yết ở nước ngoài, công ty tư vấn,
# công ty con trùng mã.
AMBIGUOUS_IN_ARTICLE = {
    "VNM": re.compile(r"VNM ETF|Vietnam ETF|Market Vectors"),
    "VNL": re.compile(r"(?i)vinaland|vinacapital"),
    "VNI": re.compile(r"(?i)vinaland|vinacapital"),
    "BCG": re.compile(r"(?i)boston consulting|tư vấn boston"),
    "MSB": re.compile(r"Masan Beverage"),
}

# Luật 1.2 - Mã đứng trơn: 3 ký tự viết hoa có trong danh sách mã, không dính chữ/dấu chấm hai
# bên. Mỗi mã tìm được xét tiếp ngữ cảnh ngay trước/sau nó (luật 1.3, 1.4) - chỉ cần xem vài ký
# tự quanh mã, không cắt cả đoạn từ đầu bài.
BARE_SYMBOL = re.compile(r"(?<![\w.])([A-Z][A-Z0-9]{2})(?![\w])")
CONTEXT_CHARS = 20

# Luật 1.3 - Ngữ cảnh xác nhận mã: bằng chứng chắc chắn là mã, dùng để xác nhận mã dễ nhầm (luật 1.1).
# - Tên sàn ngay trước ("HOSE: CEO"): tên sàn không phân biệt hoa/thường ("HoSE", "UpCoM"...).
#   Không có "OTC": thị trường tự do không phải sàn niêm yết, mã sau "OTC:" là mã đứng trơn.
# - "Chứng khoán"/"CTCK" ngay trước ("Chứng khoán VFS", "CTCK FPT"): chỉ nhận khi mã là của một
#   công ty chứng khoán; mã khác sau các chữ này thì bỏ. "CTCK FPT (FPTS)" là Chứng khoán FPT
#   (FTS), không phải mã FPT; "Công ty chứng khoán SJC" không phải Sông Đà 1.01 (SJC); "sàn
#   chứng khoán SSI" vẫn là SSI. Không áp dụng cho "mã chứng khoán BID" (đó là cách gọi mã).
EXCHANGE_BEFORE = re.compile(r"\b(?i:hose|hsx|hnx|upcom)\s*:\s*\Z")
SECURITIES_BEFORE = re.compile(r"(?<!\w)(?<!mã )(?:chứng khoán|ctck)\s*$", re.IGNORECASE)

# Luật 1.4 - Ngữ cảnh không phải mã: bỏ mã khi chữ ngay trước/sau cho thấy chắc chắn là nghĩa
# khác. Mỗi dòng: (mẫu khớp ở cuối đoạn ngay trước mã, mẫu khớp ở đầu đoạn ngay sau mã, chỉ áp
# dụng cho các mã này - None là mọi mã). Thêm trường hợp mới bằng 1 dòng ở đây.
NOT_SYMBOL_CONTEXTS = [
    # Thành phố: "TP. HCM"
    (re.compile(r"(?:TP|Tp|tp)\.?\s*$"), None, None),
    # Đơn vị tiền: "2,1 tỷ VND", tỷ giá "VND/USD"
    (re.compile(r"(?:\d|tỷ|triệu|nghìn|ngàn)\s*$"), None, {"VND"}),
    (None, re.compile(r"/"), {"VND"}),
    # Số hiệu văn bản: "Thông tư 64/2010/TT-BTC", "Văn bản 8058/BTC-CST", "TT-BTC" - mã dính liền
    # sau cụm bắt đầu bằng số và "/", hoặc sau tiền tố loại văn bản + gạch nối. Không bỏ mọi mã
    # dính gạch nối: "Sacombank-SBS", "cặp STB-EIB" vẫn là công ty. Dùng \Z chứ không dùng $: $
    # khớp cả trước "\n" cuối, khiến ngày cuối dòng "30/9/2011\nSSC" bị coi là số hiệu.
    (re.compile(r"(?:(?<!\S)\d+/\S*|(?<![\w-])(?:TT|TTLT|QĐ|NĐ|NQ|CT|CV|TB)-)\Z"), None, None),
    # Học vị: "PGS.TS", "PGS. TS", "PGS-TS", "PGS.TSKH" là Phó giáo sư.
    (None, re.compile(r"\s*[.\-–]?\s*TS(?:KH)?\b"), None),
    # Tên công ty con viết hoa chứa mã công ty mẹ: "PV GAS D" (PGD), "PV GAS SOUTH" (PGS)
    (None, re.compile(r"\s+(?:D|SOUTH)\b"), {"GAS"}),
    # Quỹ ETF trên sàn Mỹ trùng mã Vinamilk: "VNM ETF", "Market Vectors Vietnam ETF (VNM)"
    (None, re.compile(r"\s+ETF\b"), {"VNM"}),
    (re.compile(r"ETF\s*\(\s*\Z"), None, {"VNM"}),
]


def not_a_symbol(token: str, before: str, after: str) -> bool:
    """Luật 1.4: ngữ cảnh quanh mã cho thấy đây không phải mã chứng khoán."""
    return any((only is None or token in only)
               and (before_pattern is None or before_pattern.search(before))
               and (after_pattern is None or after_pattern.match(after))
               for before_pattern, after_pattern, only in NOT_SYMBOL_CONTEXTS)


def find_symbols_in_text(text: str) -> list[tuple[str, str, str]]:
    """Mã chứng khoán viết trực tiếp trong bài -> [(mã, đoạn văn bản khớp, kiểu)].

    Kiểu: "sàn" / "chứng khoán" (luật 1.3, "HOSE: CEO", "Chứng khoán VFS"), "trơn" (luật 1.2,
    mã đứng một mình), "mơ hồ" (mã trong AMBIGUOUS_SYMBOLS, hoặc AMBIGUOUS_IN_ARTICLE khi bài có
    cụm chữ tương ứng, đứng một mình - extract_symbols_offline chỉ nhận khi bài có mã đó kiểu
    "sàn"/"chứng khoán" hoặc có tên công ty của mã).
    """
    symbols = load_symbols()[0]
    ambiguous = AMBIGUOUS_SYMBOLS | {s for s, pattern in AMBIGUOUS_IN_ARTICLE.items() if pattern.search(text)}
    found = []
    for m in BARE_SYMBOL.finditer(text):
        token = m.group(1)
        if token not in symbols:
            continue
        before = text[max(0, m.start() - CONTEXT_CHARS):m.start()]
        after = text[m.end():m.end() + CONTEXT_CHARS]
        if not_a_symbol(token, before, after):
            continue
        exchange = EXCHANGE_BEFORE.search(before)
        if exchange:
            found.append((token, exchange.group(0) + token, "sàn"))
            continue
        securities = SECURITIES_BEFORE.search(before)
        if securities:
            if token in securities_symbols():
                found.append((token, securities.group(0).strip() + " " + token, "chứng khoán"))
            continue
        found.append((token, token, "mơ hồ" if token in ambiguous else "trơn"))
    return found


# ===========================================================================
# PHẦN 2 - LUẬT CHO TÊN CÔNG TY VIẾT TRONG BÀI
# ===========================================================================

# --- 2.1 Chuẩn hoá tên (áp dụng cho cả tên trong danh sách lẫn nội dung bài) --------------

# Dấu thanh kiểu cũ/mới ở cuối âm tiết: "thuỷ" = "thủy", "hoà" = "hòa" -> đưa về 1 kiểu
# (có phụ âm cuối như "đoàn", "hoạt" thì chỉ có 1 cách viết, không đổi)
TONE_PLACEMENT = {
    old: new
    for vowel_pair, marked in [("oa", "òóỏõọ"), ("oe", "òóỏõọ"), ("uy", "ùúủũụ")]
    for old, new in zip(
        (vowel_pair[0] + ch for ch in {"oa": "àáảãạ", "oe": "èéẻẽẹ", "uy": "ỳýỷỹỵ"}[vowel_pair]),
        (ch + vowel_pair[1] for ch in marked),
    )
}
TONE_PLACEMENT_RE = re.compile("(?:" + "|".join(TONE_PLACEMENT) + r")(?!\w)")

# Viết tắt trong tên công ty -> viết đầy đủ như tên trong vn_stock_symbols.csv. Chỉ thay khi
# đứng thành 1 chữ riêng ("tmcp", "đtck" không bị đụng tới).
# - Ngành nghề: "CTCP TV-TM-DV Địa ốc Hoàng Quân" = "CTCP Tư vấn - Thương mại - Dịch vụ Địa ốc
#   Hoàng Quân" (HQC).
# - "CTCK Rồng Việt" = "Công ty Chứng khoán Rồng Việt" (VDS).
BUSINESS_ABBREVIATIONS = {
    "tv": "tư vấn",
    "tm": "thương mại",
    "dv": "dịch vụ",
    "sx": "sản xuất",
    "xnk": "xuất nhập khẩu",
    "xk": "xuất khẩu",
    "xd": "xây dựng",
    "vlxd": "vật liệu xây dựng",
    "đt": "đầu tư",
    "pt": "phát triển",
    "kd": "kinh doanh",
    "bđs": "bất động sản",
    "ctck": "cty chứng khoán",
    "cn": "công nghiệp",
    "tb": "thiết bị",
    "tct": "tổng công ty",
    "bh": "bảo hiểm",
    "nh": "ngân hàng",
    "ck": "chứng khoán",
    # Địa danh, tên tập đoàn và chính tả viết khác nhau giữa bài báo và tên trong danh sách:
    # "CTCP Kim khí TP.HCM" = "... Thành phố Hồ Chí Minh", "Than Hà Tu - TKV" = "... - Vinacomin"
    "tp": "thành phố",
    "hcm": "hồ chí minh",
    "tphcm": "thành phố hồ chí minh",
    "vn": "việt nam",
    "tkv": "vinacomin",
    "mĩ": "mỹ",
    "ôtô": "ô tô",
}

# Viết tắt loại hình doanh nghiệp về 1 dạng, rồi bỏ khi so "phần lõi" của tên
LEGAL_ALIASES = [
    (r"tổng công ty cổ phần", "tổng ctcp"),
    (r"công ty cổ phần", "ctcp"),
    (r"(?:công ty|cty) cp", "ctcp"),
    (r"ngân hàng thương mại cổ phần", "ngân hàng tmcp"),
    (r"tổng công ty", "tổng cty"),
    (r"công ty", "cty"),
]
LEGAL_PREFIX = re.compile(r"^(?:(?:tổng ctcp|tổng cty|ctcp|cty|tập đoàn|ngân hàng tmcp|ngân hàng)\s+)+")
LEGAL_SUFFIX = re.compile(r"(?:\s+(?:ctcp|jsc))+$")
CORE_MARKERS = ("tổng", "ngân hàng")


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFC", name).lower().replace("&", " và ")
    s = TONE_PLACEMENT_RE.sub(lambda m: TONE_PLACEMENT[m.group(0)], s)
    s = " ".join(BUSINESS_ABBREVIATIONS.get(w, w) for w in re.sub(r"[^\w\s]", " ", s).split())
    for pattern, short in LEGAL_ALIASES:
        s = re.sub(rf"\b{pattern}\b", short, s)
    return s


def _alias_sequences() -> list[tuple[list[tuple[str, ...]], list[str]]]:
    """LEGAL_ALIASES dạng dãy chữ: "(?:công ty|cty) cp" -> [("công","ty","cp"), ("cty","cp")]."""
    sequences = []
    for pattern, short in LEGAL_ALIASES:
        m = re.fullmatch(r"\(\?:([^)]*)\)\s*(.*)", pattern)
        alternatives = [a.split() + m.group(2).split() for a in m.group(1).split("|")] if m else [pattern.split()]
        sequences.append(([tuple(a) for a in alternatives], short.split()))
    return sequences


LEGAL_ALIAS_SEQUENCES = _alias_sequences()
NAME_SPACE = re.compile(r"[ \t\xa0]+")
# Chữ viết hoa đứng sau tên nhưng không làm thành tên công ty khác ("CTCP Gelex Group", "... JSC")
NAME_TAIL_WORDS = {"jsc", "corp", "corporation", "group", "holding", "holdings", "co", "company"}


class WordInfo(NamedTuple):
    """Thông tin chữ gốc của 1 chữ đã chuẩn hoá (name_words)."""
    cap: bool  # chữ gốc viết hoa chữ đầu
    joined: bool  # đứng liền chữ trước: chỉ cách bằng khoảng trắng, không xuống dòng, không dấu câu
    code: bool  # chữ gốc là mã CK
    inner: bool  # chữ thứ 2 trở đi của 1 chữ gốc được viết ra nhiều chữ ("TPHCM", "CTCP")
    paren: bool = False  # có dấu "(" giữa chữ trước và chữ này

    @property
    def continues_name(self) -> bool:
        """Chữ viết hoa nối liền chữ trước, không phải mã CK: vẫn thuộc cùng một tên riêng."""
        return self.cap and self.joined and not self.code and not self.inner


def name_words(text: str) -> tuple[list[str], list[WordInfo]]:
    """(các chữ của bài sau chuẩn hoá - giống normalize_name(text).split(), WordInfo từng chữ)."""
    s = unicodedata.normalize("NFC", text).replace("&", " và ")
    codes = load_symbols()[0]
    words, info, prev_end = [], [], None
    for m in re.finditer(r"\w+", s):
        raw = m.group(0)
        gap = s[prev_end:m.start()] if prev_end is not None else ""
        joined = prev_end is not None and NAME_SPACE.fullmatch(gap) is not None
        prev_end = m.end()
        low = TONE_PLACEMENT_RE.sub(lambda t: TONE_PLACEMENT[t.group(0)], raw.lower())
        for k, word in enumerate(BUSINESS_ABBREVIATIONS.get(low, low).split()):
            words.append(word)
            info.append(WordInfo(raw[0].isupper(), joined or k > 0, raw in codes, k > 0, "(" in gap and k == 0))
    for alternatives, short in LEGAL_ALIAS_SEQUENCES:  # như re.sub từng mẫu, trái sang phải
        out_words, out_info, i = [], [], 0
        while i < len(words):
            alt = next((a for a in alternatives if tuple(words[i:i + len(a)]) == a), None)
            if alt:
                out_words += short
                out_info += [info[i]] + [info[i]._replace(joined=True, inner=True, paren=False)] * (len(short) - 1)
                i += len(alt)
            else:
                out_words.append(words[i])
                out_info.append(info[i])
                i += 1
        words, info = out_words, out_info
    return words, info


EXCHANGE_WORDS = {"hose", "hsx", "hnx", "upcom"}


def tagged_code(words: list[str], info: list[WordInfo], end: int) -> str | None:
    """Mã CK kèm tên sàn trong ngoặc ngay sau tên kết thúc ở vị trí end: "CTCP X (HNX: ABC)" -> "ABC".

    Ngoặc không có tên sàn thường là tên viết tắt, có thể trùng mã khác: "Công ty Chứng khoán
    BIDV (BSC)" là BSI, "Tập đoàn Công nghiệp Cao su Việt Nam (VRG)" là GVR - không tính.
    """
    if end + 1 >= len(words) or not info[end].paren or words[end] not in EXCHANGE_WORDS:
        return None
    k = end + 1
    if info[k].code and re.fullmatch(r"[a-z0-9]{3}", words[k]):
        return words[k].upper()
    return None


def name_core(normalized: str) -> str:
    """Tên bỏ loại hình doanh nghiệp: 'ctcp tập đoàn hòa phát' -> 'hòa phát'.

    Giữ chữ "tổng" / "ngân hàng": "Tổng CTCP Đầu tư Phát triển Xây dựng" (DIG) khác
    "CTCP Đầu tư và Phát triển Xây dựng" (ING); "Ngân hàng TMCP Hàng Hải" (MSB) khác
    "Tổng Công ty Hàng hải" (MVN).
    """
    prefix = LEGAL_PREFIX.match(normalized)
    prefix = prefix.group(0) if prefix else ""
    core = LEGAL_SUFFIX.sub("", normalized[len(prefix):]).strip()
    marker = next((m for m in CORE_MARKERS if m in prefix), None)
    return f"{marker} {core}" if marker else core


# --- 2.2 Các dạng tên được nhận ----------------------------------------------------------
# Tên đầy đủ; lõi tên có loại hình đứng trước ("CTCP/Cty/Tập đoàn" + lõi); tên phụ; thương
# hiệu 1 chữ (luật dưới); tên ngắn của công ty chứng khoán (luật dưới).

BRAND_CORE = re.compile(r"^[a-z][a-z0-9]{4,}$")  # 1 chữ không dấu, >= 5 ký tự: idico, viglacera...
# Thương hiệu 1 chữ trùng chữ thông dụng / tên khác: "trang" (TFC), "green" (GPC),
# "sunshine" (KSF, lẫn Sunshine Homes), "mirae" (KMR, lẫn Chứng khoán Mirae Asset),
# "hanel" (HNE, lẫn Hanel Xốp Nhựa HNP và các công ty khác trong nhóm Hanel),
# "alphanam" (ALP, lẫn Alphanam E&C, Alphanam Cơ điện)
BRAND_STOPWORDS = {"trang", "green", "sunshine", "mirae", "hanel", "alphanam"}

# Tên ngắn của công ty chứng khoán: "Chứng khoán Nhất Việt" = CTCP Chứng khoán Nhất Việt (VFS),
# không cần "CTCP"/"Cty" đứng trước. Không nhận khi chữ ngay trước thuộc cơ quan/thị trường:
# "Sở Giao dịch Chứng khoán TP.HCM" không phải Chứng khoán TP.HCM (HCM), "Trung tâm Lưu ký
# Chứng khoán", "Ủy ban Chứng khoán", "thị trường chứng khoán", "sàn chứng khoán", "quỹ đầu tư
# chứng khoán".
SECURITIES_SHORT_NAME_BLOCKED_AFTER = {"dịch", "ký", "ban", "trường", "sàn", "tư"}

# Tên thương hiệu (cột brand_name của vn_stock_symbols.csv, nhiều tên cách nhau " | "): tên báo
# hay dùng thay cho tên pháp lý - "Techcombank" (TCB), "Thế giới Di động" (MWG), "Sabeco" (SAB).
# Gồm danh sách tuyển tay và tên thương hiệu tra trên web cho từng mã, đã đo trên kho tin: bỏ
# tên khớp từ 5 bài trở lên mà dưới 50% số bài đó có ghi mã của công ty. Không đưa vào:
# - tên tập đoàn có nhiều công ty con niêm yết mang cùng tên: Petrolimex, Vinaconex,
#   Viglacera, Vinatex ("Viglacera Tiên Sơn" là VIT, không phải VGC);
# - thương hiệu công ty chứng khoán (HSC, VCSC, FPTS, VNDirect...): trong bài thường là
#   nguồn phân tích ("Theo HSC, ...") chứ không phải công ty được nói tới;
# - tên trùng địa danh / chữ thường: Hoa Sen, Kinh Bắc, Bình Minh, Hùng Vương, Sao Mai; tên
#   tra trên web đúng 2 chữ tiếng Việt có dấu (hay trùng tên đường, tên người: Thái Hòa).
# Thương hiệu ngay sau "chứng khoán"/"bảo hiểm" là công ty con, không phải công ty mẹ:
# "Chứng khoán BIDV" (BSI), "Bảo hiểm BIDV" (BIC) không phải BID - trừ khi chính công ty của
# thương hiệu cùng ngành đó ("Bảo hiểm PJICO" vẫn là PGI). Chữ đứng trước -> ngành.
BRAND_BLOCKED_AFTER = {"khoán": "chứng khoán", "hiểm": "bảo hiểm"}


@lru_cache(maxsize=1)
def load_brands() -> dict[tuple[str, ...], tuple[str, ...]]:
    """Dãy chữ (đã chuẩn hoá như tên công ty) của thương hiệu -> các mã mang thương hiệu đó
    (nhiều mã khi 1 công ty có 2 mã)."""
    raw = pd.read_csv(SYMBOLS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    brands = raw.assign(brand_name=raw["brand_name"].str.split("|")).explode("brand_name")
    brands = brands[brands["brand_name"].str.strip() != ""]
    symbols: dict[tuple[str, ...], set[str]] = {}
    for symbol, brand in zip(brands["symbol"], brands["brand_name"]):
        symbols.setdefault(tuple(normalize_name(brand).split()), set()).add(symbol)
    return {words: tuple(sorted(s)) for words, s in symbols.items()}


@lru_cache(maxsize=1)
def name_patterns() -> tuple[dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]], frozenset]:
    """(chữ đầu -> [(dãy chữ cần khớp, các mã)], tập dãy chữ là tên ngắn công ty chứng khoán).

    Dãy dài xếp trước để ưu tiên khớp dài nhất.
    """
    _, by_full, by_core, by_alias = load_symbols()
    patterns: dict[tuple[str, ...], tuple[str, ...]] = {}
    for name, symbol in by_full.items():
        patterns[tuple(name.split())] = symbol
    for core, symbol in by_core.items():
        if core.startswith("tổng "):
            forms = [f"tổng ctcp {core[5:]}", f"tổng cty {core[5:]}"]
        elif core.startswith("ngân hàng "):
            forms = [core, f"ngân hàng tmcp {core[10:]}"]
        else:
            forms = [f"ctcp {core}", f"cty {core}", f"tập đoàn {core}"]
        for form in forms:
            patterns.setdefault(tuple(form.split()), symbol)
    for alias, symbol in by_alias.items():
        for form in (f"ctcp {alias}", f"cty {alias}", f"tập đoàn {alias}"):
            patterns.setdefault(tuple(form.split()), symbol)
    # Thương hiệu 1 chữ chỉ lấy từ tên hiện tại: tên cũ 1 chữ có thể đã thành thương hiệu của
    # công ty khác ("CTCP Vincom" là VIC trước 2013, nay "Vincom" là TTTM của Vincom Retail - VRE)
    raw = pd.read_csv(SYMBOLS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    current = {name_core(normalize_name(n)) for n in raw["organ_name"].str.split("|").explode()}
    current |= {c[len("tổng "):] for c in current if c.startswith("tổng ")}  # "Tổng công ty IDICO" -> idico
    for core, symbol in {**by_core, **by_alias}.items():
        if BRAND_CORE.match(core) and core not in BRAND_STOPWORDS and core in current:
            patterns.setdefault((core,), symbol)
    securities_short = set()
    for core, symbol in by_core.items():
        words = tuple(core.split())
        if words[:2] == ("chứng", "khoán") and len(words) > 2 and words not in patterns:
            patterns[words] = symbol
            securities_short.add(words)
    for words, symbol in load_brands().items():
        patterns.setdefault(words, symbol)

    by_first: dict[str, list] = {}
    for words, symbol in sorted(patterns.items(), key=lambda p: -len(p[0])):
        by_first.setdefault(words[0], []).append((words, symbol))
    return by_first, frozenset(securities_short)


def find_names_in_text(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Tên công ty (theo vn_stock_symbols.csv) xuất hiện trong bài -> [(các mã, tên đã khớp)].

    Thường mỗi tên 1 mã; tên dùng chung cho nhiều mã (load_symbols) trả về đủ các mã đó.
    Khớp dài nhất tại mỗi vị trí; bỏ khi tên còn tiếp tục, vì đó là tên dài hơn không có trong
    file: chữ ngay sau là số ("CTCP Sông Đà 25" không được tính là "CTCP Sông Đà"), hoặc là chữ
    viết hoa nối liền tên (name_words) - "CTCP Thủy sản Cửu Long An Giang" (ACL) không phải
    "CTCP Thủy sản Cửu Long" (CLP), "CTCP Viglacera Đông Anh" (DAC) không phải Viglacera (VGC).
    Không áp dụng cho thương hiệu ("Techcombank Hà Nội" vẫn là TCB) và chữ đuôi trong
    NAME_TAIL_WORDS ("CTCP X Group"); không
    khớp tên công ty thường ngay sau chữ "tổng" ("Tổng CTCP Đầu tư và Phát triển Xây
    dựng" không phải ING); tên ngắn công ty chứng khoán không khớp sau các chữ trong
    SECURITIES_SHORT_NAME_BLOCKED_AFTER; thương hiệu không khớp sau các chữ trong
    BRAND_BLOCKED_AFTER; tên ngay sau có mã trong ngoặc của công ty khác thì bỏ, bài đã ghi rõ
    công ty nào ("CTCP Chứng khoán Sài Gòn (HNX: SHS)" không phải SSI). Cụm tên không khớp chính xác tên nào thì thử tên rút gọn
    (find_short_names), tên đã khớp ghi "rút gọn: ...".
    """
    by_first, securities_short = name_patterns()
    brands = load_brands()
    words, info = name_words(text)
    found, covered, i = [], [False] * len(words), 0
    while i < len(words):
        previous = words[i - 1] if i > 0 else ""
        for pattern, symbols in by_first.get(words[i], ()):
            end = i + len(pattern)
            if previous == "tổng" and pattern[0] != "tổng":
                continue
            if pattern in securities_short and previous in SECURITIES_SHORT_NAME_BLOCKED_AFTER:
                continue
            if (pattern in brands and previous in BRAND_BLOCKED_AFTER
                    and not set(symbols) <= sector_symbols(BRAND_BLOCKED_AFTER[previous])):
                continue
            if tuple(words[i:end]) == pattern:
                longer = end < len(words) and (
                    words[end].isdigit()
                    or (info[end].continues_name and pattern not in brands and words[end] not in NAME_TAIL_WORDS))
                tag = tagged_code(words, info, end)
                if not longer and (tag is None or tag in symbols):
                    found.append((symbols, " ".join(pattern)))
                    covered[i:end] = [True] * (end - i)
                i = end - 1
                break
        i += 1
    return found + find_short_names(words, info, covered)


# --- 2.3 Tên rút gọn ----------------------------------------------------------------------
# Báo hay bỏ bớt chữ của tên chính thức: "CTCP Thủy sản Cửu Long An Giang" = CTCP Xuất nhập khẩu
# Thủy sản Cửu Long An Giang (ACL), "CTCP Than Hà Tu" = CTCP Than Hà Tu - Vinacomin (THT). Thay vì
# thêm từng cách viết vào danh sách, cắt cụm tên trong bài (chữ loại hình + các chữ viết hoa liền
# nhau, chữ thường chỉ được kẹp giữa 2 chữ viết hoa) rồi nhận khi cụm là tên chính thức bỏ bớt
# chữ, giữ đúng thứ tự, của đúng 1 công ty. Cụm phải >= SHORT_NAME_MIN_WORDS chữ và có chữ riêng
# (không thuộc chữ chung - có trong > SHORT_NAME_COMMON_RATE số tên công ty). Chữ được bỏ theo
# vị trí, đúng cách báo rút gọn tên:
# - đầu tên: chữ chung về ngành nghề ("Xuất nhập khẩu Thủy sản Cửu Long An Giang");
# - cuối tên: đuôi tập đoàn / quốc gia SHORT_NAME_TAIL_WORDS ("Than Hà Tu - Vinacomin");
# - giữa tên: tên tập đoàn SHORT_NAME_GROUP_WORDS ("Xi măng Vicem Hoàng Mai").
# Bỏ chữ khác ở cuối là tên bị cắt cụt ("Đại lý Hàng" của "Đại lý Hàng hải"), bỏ số là công ty
# khác (Nhơn Trạch 2). Cụm nằm trong tên của >= 2 công ty (bỏ chữ kiểu gì) thì không đoán ("Á
# Châu", "Hàng không Việt Nam"). Không nhận cụm có TNHH/MTV (không phải CTCP).
NAME_MARKERS = [("tổng", "ctcp"), ("tổng", "cty"), ("ngân", "hàng", "tmcp"), ("ngân", "hàng"), ("ctcp",),
                ("cty",), ("tập", "đoàn")]
SHORT_NAME_MIN_WORDS = 2
SHORT_NAME_COMMON_RATE = 0.005
SHORT_NAME_GROUP_WORDS = {"vinacomin", "vicem", "vnsteel", "petrolimex"}
SHORT_NAME_TAIL_WORDS = SHORT_NAME_GROUP_WORDS | {"việt", "nam"}


@lru_cache(maxsize=1)
def short_name_index() -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], dict[str, list[int]], frozenset]:
    """([(dãy chữ lõi tên, các mã)], chữ -> vị trí các lõi có chữ đó, tập chữ chung)."""
    sym = read_symbol_names()
    entries: dict[tuple[str, ...], set] = {}
    for symbol, name in zip(sym["symbol"], sym["organ_name"]):
        core = name_core(normalize_name(name)).split()
        while core and core[0] in {"tổng", "ngân", "hàng", "tmcp", "tập", "đoàn"}:  # chữ loại hình đầu lõi
            core = core[1:]
        if core:
            entries.setdefault(tuple(core), set()).add(COMPANY_GROUP_OF.get(symbol, (symbol,)))
    entries_list = [(core, groups) for core, groups in entries.items()]
    df = Counter(w for core, _ in entries_list for w in set(core))
    common = frozenset(w for w, n in df.items() if n > SHORT_NAME_COMMON_RATE * len(entries_list))
    by_word: dict[str, list[int]] = {}
    for k, (core, _) in enumerate(entries_list):
        for w in set(core):
            by_word.setdefault(w, []).append(k)
    return [(core, tuple(sorted(groups))) for core, groups in entries_list], by_word, common


def find_short_names(words: list[str], info: list[WordInfo], covered: list[bool]) -> list[tuple[tuple[str, ...], str]]:
    """Tên rút gọn trong các cụm tên chưa khớp chính xác (xem 2.3) -> [(các mã, "rút gọn: <cụm>")]."""
    entries, by_word, common = short_name_index()
    found, i, n = [], 0, len(words)
    while i < n:
        marker = next((m for m in NAME_MARKERS if tuple(words[i:i + len(m)]) == m), None)
        if not marker or covered[i]:
            i += 1
            continue
        start = end = i + len(marker)
        while end < n and not covered[end] and not info[end].code and (end == start or info[end].joined):
            if any(tuple(words[end:end + len(m)]) == m for m in NAME_MARKERS):
                break
            if info[end].inner or info[end].cap or words[end].isdigit():
                end += 1
            elif end > start and end + 1 < n and info[end + 1].joined and info[end + 1].cap and not info[end + 1].code:
                end += 1  # chữ thường kẹp giữa tên: "Thủy sản Cửu Long"
            else:
                break
        span = words[start:end]
        # Cụm dừng trước 1 tên khác đã khớp mà vẫn viết liền ("CTCP Giám định Vinacontrol" dừng
        # trước thương hiệu Vinacontrol) thì không phải tên rút gọn
        continued = end < n and info[end].continues_name and words[end] not in NAME_TAIL_WORDS
        tag = tagged_code(words, info, end)
        if (end > start and info[start].cap and len(span) >= SHORT_NAME_MIN_WORDS
                and not continued and not {"tnhh", "mtv"} & set(span)):
            distinct = [w for w in span if w not in common and not w.isdigit()]
            if distinct:
                rare = min(distinct, key=lambda w: len(by_word.get(w, ())))
                containing = [k for k in by_word.get(rare, ()) if _dropped_words(span, entries[k][0]) is not None]
                groups = {g for k in containing for g in entries[k][1]}
                if (len(groups) == 1 and (tag is None or tag in next(iter(groups)))
                        and any(_short_name_ok(span, entries[k][0], common) for k in containing)):
                    found.append((next(iter(groups)), "rút gọn: " + " ".join(span)))
        i = max(end, i + 1)
    return found


def _dropped_words(span: list[str], core: tuple[str, ...]) -> tuple[list[str], list[str], list[str]] | None:
    """Chữ của core bị bỏ để còn span (giữ thứ tự): (đầu, giữa, cuối); None nếu span không nằm trong core."""
    positions, j = [], 0
    for k, word in enumerate(core):
        if j < len(span) and word == span[j]:
            positions.append(k)
            j += 1
    if j < len(span):
        return None
    first, last = positions[0], positions[-1]
    middle = [core[k] for k in range(first, last + 1) if k not in positions]
    return list(core[:first]), middle, list(core[last + 1:])


def _short_name_ok(span: list[str], core: tuple[str, ...], common: frozenset) -> bool:
    """span là tên rút gọn hợp lệ của core: chữ bị bỏ đúng vị trí, đúng loại (xem 2.3)."""
    head, middle, tail = _dropped_words(span, core)
    return (bool(head or middle or tail)
            and all(w in common and not w.isdigit() for w in head)
            and all(w in SHORT_NAME_GROUP_WORDS for w in middle)
            and all(w in SHORT_NAME_TAIL_WORDS for w in tail))


# ===========================================================================
# PHẦN 3 - KẾT HỢP MÃ VÀ TÊN THÀNH KẾT QUẢ CỦA 1 BÀI
# ===========================================================================

SHARED_NAME_EXCLUDED = "loại: tên dùng chung nhiều mã, bài không ghi mã"


def extract_symbols_offline(title, description, content) -> list[dict]:
    """Mã chứng khoán của các công ty được nhắc trong bài (title + sapo + content).

    Kiểm tra cả mã viết trực tiếp trong bài (PHẦN 1) lẫn tên công ty (PHẦN 2); với mỗi công
    ty, ưu tiên mã: bài có ghi mã thì ghi nhận theo mã ("mã CK"), chỉ nhắc tên thì lấy mã qua
    tên ("tên công ty"). Mã mơ hồ đứng trơn chỉ được nhận khi bài cũng có mã đó kèm tên sàn,
    có mã đó ngay sau chữ "Chứng khoán", hoặc có tên công ty của mã. Trả về
    [{"symbol", "matched_by", "evidence", "position", "mentions"}], mỗi mã 1 lần:
    matched_by "mã CK" / "tên công ty" / "tên rút gọn" / "thương hiệu" / "cùng công ty"; position là chỗ
    đầu tiên mã/tên xuất hiện ("tiêu đề", "sapo" = description, "nội dung"); mentions là số
    lần mã/tên của công ty được nhắc trong cả bài - để bước lọc equity news biết công ty là
    chủ thể của bài hay chỉ được nhắc qua.

    Tên dùng chung cho nhiều công ty ("CTCP Nam Việt" = ANV và NAV) không tự ra mã: bài có
    ghi một trong các mã đó thì đã khớp qua mã; bài nhắc tên mà không ghi mã nào trong số đó
    thì chỉ bỏ tên đó, các mã khác của bài vẫn giữ ("VHC ... so với CTCP Nam Việt" -> VHC).
    Chỉ khi bài không còn mã nào thì loại cả bài - trả về đúng 1 dòng {"symbol": "",
    "matched_by": SHARED_NAME_EXCLUDED, "evidence": "ctcp nam việt (ANV/NAV)"} để đưa sang
    file review.

    Nhóm cùng công ty trong SAME_COMPANY_SYMBOLS (SIC = ANI): khớp ra 1 mã thì thêm các mã
    còn lại của nhóm, matched_by "cùng công ty".
    """
    text = "\n".join(x for x in (title, description, content) if isinstance(x, str) and x.strip())
    direct = find_symbols_in_text(text)
    names = find_names_in_text(text)
    company_names = [(symbols, evidence) for symbols, evidence in names
                     if len(symbols) == 1 or symbols in SAME_COMPANY_GROUPS]
    shared_names = [(symbols, evidence) for symbols, evidence in names
                    if len(symbols) > 1 and symbols not in SAME_COMPANY_GROUPS]
    # Mã mơ hồ được xác nhận bởi: mã kèm tên sàn, mã ngay sau "Chứng khoán", tên công ty của mã.
    # Tên dùng chung cũng xác nhận mã mơ hồ thuộc nhóm của nó: "CTCP Nam Việt (NAV)" -> NAV
    confirmed = ({symbol for symbol, _, kind in direct if kind in ("sàn", "chứng khoán")}
                 | {symbol for symbols, _ in company_names + shared_names for symbol in symbols})

    found: dict[str, dict] = {}
    # Mã ghi trong bài trước (kể cả mã mơ hồ đã xác nhận), rồi mới đến tên công ty: công ty
    # có cả mã lẫn tên thì được ghi nhận theo mã. Bằng chứng ưu tiên mã kèm tên sàn ("HOSE: AGF").
    for symbol, evidence, kind in sorted(direct, key=lambda d: d[2] != "sàn"):
        if kind != "mơ hồ" or symbol in confirmed:
            found.setdefault(symbol, {"symbol": symbol, "matched_by": "mã CK", "evidence": evidence})
    brands = load_brands()
    for symbols, evidence in company_names:
        matched_by = ("thương hiệu" if tuple(evidence.split()) in brands
                      else "tên rút gọn" if evidence.startswith("rút gọn: ") else "tên công ty")
        for symbol in symbols:
            found.setdefault(symbol, {"symbol": symbol, "matched_by": matched_by, "evidence": evidence})
    for symbol in list(found):
        for other in COMPANY_GROUP_OF.get(symbol, ()):
            found.setdefault(other, {"symbol": other, "matched_by": "cùng công ty", "evidence": symbol})
    if not found:
        # Không còn mã nào: nếu bài có nhắc tên dùng chung thì đưa sang review kèm tên đó
        for symbols, evidence in shared_names:
            return [{"symbol": "", "matched_by": SHARED_NAME_EXCLUDED,
                     "evidence": f"{evidence} ({'/'.join(symbols)})"}]

    mentions = Counter(symbol for symbol, _, _ in direct)
    mentions.update(symbol for symbols, _ in names for symbol in symbols)
    in_title, in_sapo = symbols_in_part(title), symbols_in_part(description)
    for symbol, item in found.items():
        # "cùng công ty" (ANI thêm từ SIC): lấy vị trí và số lần nhắc của mã đã khớp
        source = item["evidence"] if item["matched_by"] == "cùng công ty" else symbol
        item["position"] = "tiêu đề" if source in in_title else "sapo" if source in in_sapo else "nội dung"
        item["mentions"] = mentions[source]
    return list(found.values())


def symbols_in_part(text) -> set[str]:
    """Các mã nhận ra trong 1 phần của bài (tiêu đề hoặc sapo), để biết vị trí của mã.

    Không xét lại luật mã mơ hồ: mã đã được extract_symbols_offline nhận ở cấp cả bài.
    """
    if not isinstance(text, str) or not text.strip():
        return set()
    return ({symbol for symbol, _, _ in find_symbols_in_text(text)}
            | {symbol for symbols, _ in find_names_in_text(text) for symbol in symbols})
