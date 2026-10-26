"""Tách mã chứng khoán cho từng bài tin (offline, theo vn_stock_symbols.csv).

Cấu trúc file:
  PHẦN 0 - Danh sách mã và tên công ty (đọc vn_stock_symbols.csv, nhóm cùng công ty)
  PHẦN 1 - Luật cho MÃ chứng khoán viết trong bài       -> find_symbols_in_text
  PHẦN 2 - Luật cho TÊN công ty viết trong bài           -> find_names_in_text
  PHẦN 3 - Kết hợp mã và tên thành kết quả của 1 bài     -> extract_symbols_offline
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

# ===========================================================================
# PHẦN 0 - DANH SÁCH MÃ VÀ TÊN CÔNG TY
# ===========================================================================

SYMBOLS_PATH = Path(__file__).resolve().parent / "data_news" / "vn_stock_symbols.csv"

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
    """vn_stock_symbols.csv, mỗi tên 1 dòng (1 mã có thể có nhiều tên cũ/mới, cách nhau " | ")."""
    raw = pd.read_csv(SYMBOLS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    sym = raw.assign(organ_name=raw["organ_name"].str.split("|")).explode("organ_name")
    sym["organ_name"] = sym["organ_name"].str.strip()
    return sym


@lru_cache(maxsize=1)
def load_symbols() -> tuple[frozenset, dict, dict, dict]:
    """(tập mã cổ phiếu, tên chuẩn hoá -> mã, lõi tên -> mã, tên phụ -> mã); mỗi tên -> tuple mã.

    Tên chuẩn hoá và lõi tên theo luật ở PHẦN 2 (normalize_name, name_core).
    Tên phụ: lõi bỏ chữ "tổng"/"ngân hàng" ("Tổng công ty IDICO" -> "idico"), chỉ dùng
    khi không trùng lõi của công ty khác. Nhiều công ty khác nhau có đúng cùng tên chính thức
    ("CTCP Nam Việt" = ANV và NAV, "CTCP Xây dựng Số 5" = SC5 và VC5) thì tên đó giữ đủ các
    mã - tên đó không tự ra mã được, xem extract_symbols_offline; lõi/tên phụ trùng giữa các
    công ty tên khác nhau thì bỏ, không đoán. Nhóm trong SAME_COMPANY_SYMBOLS: mọi tên của
    nhóm khớp ra cả nhóm.
    """
    sym = read_symbol_names()
    full = sym["organ_name"].map(normalize_name)  # mã rời sàn không có tên -> "", bị bỏ ở build_index
    core = full.map(name_core)
    marker = r"^(?:" + "|".join(CORE_MARKERS) + r") "
    alias = core.where(core.str.contains(marker)).str.replace(marker, "", regex=True)

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
    sym = read_symbol_names()
    return frozenset(sym.loc[sym["organ_name"].map(normalize_name).str.contains(r"\bchứng khoán\b"), "symbol"])


# ===========================================================================
# PHẦN 1 - LUẬT CHO MÃ CHỨNG KHOÁN VIẾT TRONG BÀI
# ===========================================================================

# Luật 1.1 - Mã dễ nhầm. Mã trùng từ viết tắt thông dụng (USD, MTV = "một thành viên", CEO,
# BOT, CPI...) hoặc tên công ty khác (CMC = CMC Corp nhưng mã CMC là công ty khác): mã đứng
# trơn chỉ được nhận khi bài có mã đó kèm tên sàn ("HOSE: CEO"), có mã đó ngay sau chữ
# "Chứng khoán" (luật 1.4), hoặc có tên công ty của mã (PHẦN 2). Lấy từ toàn bộ kho tin: mã
# đứng trơn >= 25 lần nhưng có tên sàn đi kèm < 5% số lần; bỏ FPT, SHB, ACV, BSR vì đứng
# trơn vẫn đúng là công ty đó.
AMBIGUOUS_SYMBOLS = {
    "AVG", "BCA", "BCR", "BIG", "BOT", "BSC", "BSL", "BWA", "CAR", "CCP", "CDH", "CEO",
    "CGV", "CHC", "CIP", "CMC", "CNC", "CNN", "CPH", "CPI", "CSI", "DAS", "DCF", "DCG",
    "DIC", "DMS", "DNA", "DNT", "DOC", "DWS", "EPC", "FCC", "FSO", "GIC", "HAC", "HDS",
    "HSV", "ICT", "MDF", "MEF", "MTA", "MTV", "NAV", "NDT", "NJC", "NO1", "PLE", "QNP",
    "SBG", "SCG", "SGI", "SPC", "SPV", "THM", "TIN", "TPS", "TTC", "USD", "VDB", "VEC",
    # VFS: Chứng khoán Nhất Việt, nhưng trong tin "VFS" còn là mã VinFast trên Nasdaq, máy thở
    # VFS-410 của Vingroup, Hãng phim truyện Việt Nam (VFS)
    "VFS", "VNX", "VPA", "VPS", "VRG", "VTM",
    # Thêm khi chuyển sang vn_stock_symbols.csv, cùng tiêu chí, đã xem ngữ cảnh trong bài:
    # AUM = tài sản quản lý, BQP = Bộ Quốc phòng (Beta-BQP), HPA = Trung tâm Xúc tiến Hà Nội,
    # SCA = quỹ SSI-SCA, TCX = The CrownX (Masan), UPS = bộ lưu điện / hãng UPS, ALC (ALC II,
    # tấm ALC), ANI (CTCP ANI, mã cũ SIC), TEC (Năng lượng TEC). Không thêm F88, CRV: đứng
    # trơn vẫn đúng là công ty đó (chỉ là chưa niêm yết trong giai đoạn của kho tin).
    "ALC", "ANI", "AUM", "BQP", "HPA", "SCA", "TCX", "TEC", "UPS",
}

# Luật 1.2 - Mã có tên sàn đi kèm. Tên sàn không phân biệt hoa/thường ("HoSE", "UpCoM"...),
# mã thì phải viết hoa. Không có "OTC": thị trường tự do không phải sàn niêm yết, mã sau
# "OTC:" được xử lý như mã đứng trơn.
EXCHANGE_SYMBOL = re.compile(r"\b(?i:hose|hsx|hnx|upcom)\s*:\s*([A-Z0-9]{3})\b")

# Luật 1.3 - Mã đứng trơn: 3 ký tự viết hoa, không dính chữ/dấu chấm hai bên; bỏ các trường
# hợp chắc chắn không phải mã
BARE_SYMBOL = re.compile(r"(?<![\w.])([A-Z][A-Z0-9]{2})(?![\w])")
CITY_BEFORE = re.compile(r"(?:TP|Tp|tp)\.?\s*$")  # "TP. HCM": HCM là thành phố, không phải mã HCM
AMOUNT_BEFORE = re.compile(r"(?:\d|tỷ|triệu|nghìn|ngàn)\s*$")  # "2,1 tỷ VND": VND là tiền, không phải mã VND

# Luật 1.4 - Mã ngay sau chữ "Chứng khoán"/"CTCK" ("Chứng khoán VFS", "CTCK FPT"): chỉ nhận khi
# mã đó là của một công ty chứng khoán, và khi đó nhận luôn cả mã dễ nhầm. "CTCK FPT (FPTS)"
# là Chứng khoán FPT (FTS), không phải mã FPT; "Công ty chứng khoán SJC" không phải Sông Đà
# 1.01 (SJC); "sàn chứng khoán SSI" vẫn là SSI. Không áp dụng cho "mã chứng khoán BID" (đó là
# cách gọi mã, không phải tên công ty chứng khoán).
SECURITIES_BEFORE = re.compile(r"(?<!\w)(?<!mã )(?:chứng khoán|ctck)\s*$", re.IGNORECASE)


def find_symbols_in_text(text: str) -> list[tuple[str, str, str]]:
    """Mã chứng khoán viết trực tiếp trong bài -> [(mã, đoạn văn bản khớp, kiểu)].

    Kiểu: "sàn" (luật 1.2, "HOSE: CEO"), "chứng khoán" (luật 1.4, "Chứng khoán VFS"), "trơn"
    (luật 1.3, mã đứng một mình), "mơ hồ" (mã trong AMBIGUOUS_SYMBOLS đứng một mình -
    extract_symbols_offline chỉ nhận khi bài có mã đó kiểu "sàn"/"chứng khoán" hoặc có tên
    công ty của mã).
    """
    symbols = load_symbols()[0]
    found = []
    for m in EXCHANGE_SYMBOL.finditer(text):
        if m.group(1) in symbols:
            found.append((m.group(1), m.group(0), "sàn"))
    for m in BARE_SYMBOL.finditer(text):
        # Chỉ cần xem vài ký tự ngay trước mã ("TP. ", "tỷ ") - không cắt cả đoạn từ đầu bài
        token, before = m.group(1), text[max(0, m.start() - 20):m.start()]
        if token not in symbols or CITY_BEFORE.search(before):
            continue
        if token == "VND" and (AMOUNT_BEFORE.search(before) or text[m.end():m.end() + 1] == "/"):
            continue
        securities = SECURITIES_BEFORE.search(before)
        if securities:
            if token in securities_symbols():
                found.append((token, securities.group(0).strip() + " " + token, "chứng khoán"))
            continue
        found.append((token, token, "mơ hồ" if token in AMBIGUOUS_SYMBOLS else "trơn"))
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
    for core, symbol in {**by_core, **by_alias}.items():
        if BRAND_CORE.match(core) and core not in BRAND_STOPWORDS:
            patterns.setdefault((core,), symbol)
    securities_short = set()
    for core, symbol in by_core.items():
        words = tuple(core.split())
        if words[:2] == ("chứng", "khoán") and len(words) > 2 and words not in patterns:
            patterns[words] = symbol
            securities_short.add(words)

    by_first: dict[str, list] = {}
    for words, symbol in sorted(patterns.items(), key=lambda p: -len(p[0])):
        by_first.setdefault(words[0], []).append((words, symbol))
    return by_first, frozenset(securities_short)


def find_names_in_text(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Tên công ty (theo vn_stock_symbols.csv) xuất hiện trong bài -> [(các mã, tên đã khớp)].

    Thường mỗi tên 1 mã; tên dùng chung cho nhiều mã (load_symbols) trả về đủ các mã đó.
    Khớp dài nhất tại mỗi vị trí; bỏ khi chữ ngay sau là số, vì đó là tên dài hơn
    không có trong file ("CTCP Sông Đà 25" không được tính là "CTCP Sông Đà"); không
    khớp tên công ty thường ngay sau chữ "tổng" ("Tổng CTCP Đầu tư và Phát triển Xây
    dựng" không phải ING); tên ngắn công ty chứng khoán không khớp sau các chữ trong
    SECURITIES_SHORT_NAME_BLOCKED_AFTER.
    """
    by_first, securities_short = name_patterns()
    words = normalize_name(text).split()
    found, i = [], 0
    while i < len(words):
        previous = words[i - 1] if i > 0 else ""
        for pattern, symbols in by_first.get(words[i], ()):
            end = i + len(pattern)
            if previous == "tổng" and pattern[0] != "tổng":
                continue
            if pattern in securities_short and previous in SECURITIES_SHORT_NAME_BLOCKED_AFTER:
                continue
            if tuple(words[i:end]) == pattern:
                if not (end < len(words) and words[end].isdigit()):
                    found.append((symbols, " ".join(pattern)))
                i = end - 1
                break
        i += 1
    return found


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
    [{"symbol", "matched_by", "evidence"}], mỗi mã 1 lần.

    Tên dùng chung cho nhiều công ty ("CTCP Nam Việt" = ANV và NAV) không tự ra mã: bài có
    ghi một trong các mã đó thì đã khớp qua mã; bài nhắc tên mà không ghi mã nào trong số đó
    thì loại cả bài - trả về đúng 1 dòng {"symbol": "", "matched_by": SHARED_NAME_EXCLUDED,
    "evidence": "ctcp nam việt (ANV/NAV)"} để đưa sang file review.

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
    # có cả mã lẫn tên thì được ghi nhận theo mã
    for symbol, evidence, kind in direct:
        if kind != "mơ hồ" or symbol in confirmed:
            found.setdefault(symbol, {"symbol": symbol, "matched_by": "mã CK", "evidence": evidence})
    for symbols, evidence in company_names:
        for symbol in symbols:
            found.setdefault(symbol, {"symbol": symbol, "matched_by": "tên công ty", "evidence": evidence})
    for symbols, evidence in shared_names:
        if not any(symbol in found for symbol in symbols):
            return [{"symbol": "", "matched_by": SHARED_NAME_EXCLUDED,
                     "evidence": f"{evidence} ({'/'.join(symbols)})"}]
    for symbol in list(found):
        for other in COMPANY_GROUP_OF.get(symbol, ()):
            found.setdefault(other, {"symbol": other, "matched_by": "cùng công ty", "evidence": symbol})
    return list(found.values())
