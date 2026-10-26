from __future__ import annotations
import ast
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from News.EDA_clean.EDA_raw_news import add_description_word_count
    from News.EDA_clean.EDA_raw_news import build_year_before_2010_rows
    from News.EDA_clean.EDA_raw_news import load_input_data
    from News.EDA_clean.EDA_raw_news import NULL_LIKE_VALUES
    from News.EDA_clean.EDA_raw_news import normalize_text_columns
    from News.EDA_clean.EDA_raw_news import parse_publication_date_series
    from News.EDA_clean.EDA_raw_news import TEXT_COLUMNS
except ImportError:
    from News.EDA_clean.EDA_raw_news import add_description_word_count
    from News.EDA_clean.EDA_raw_news import build_year_before_2010_rows
    from News.EDA_clean.EDA_raw_news import load_input_data
    from News.EDA_clean.EDA_raw_news import NULL_LIKE_VALUES
    from News.EDA_clean.EDA_raw_news import normalize_text_columns
    from News.EDA_clean.EDA_raw_news import parse_publication_date_series
    from News.EDA_clean.EDA_raw_news import TEXT_COLUMNS


DATA_NEWS_DIR = PROJECT_ROOT / "data_News"
INPUT_PATH = DATA_NEWS_DIR / "dataset_news.parquet"
OUTPUT_PATH = DATA_NEWS_DIR / "clean_news.parquet"


def normalize_domain_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    normalize_domain = out.copy()
    domain = normalize_domain["domain"].astype("string").str.lower()
    domain = domain.mask(domain.isna() | domain.fillna("").isin(NULL_LIKE_VALUES))
    domain = domain.str.replace(r"^https?://", "", regex=True)
    domain = domain.str.replace(r"^www\.", "", regex=True)
    domain = domain.str.split("/", n=1).str[0]
    domain = domain.str.split("?", n=1).str[0]
    domain = domain.str.split("#", n=1).str[0]
    normalize_domain["domain"] = domain.mask(domain.fillna("").eq("")).astype("string")
    return out, normalize_domain


def parse_dict_like_keywords(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    original_keywords = out["keywords"].astype("string")

    def parse_keyword_value(raw: object) -> object:
        if pd.isna(raw):
            return pd.NA

        if isinstance(raw, dict):
            keys = [str(key) for key in raw.keys() if str(key) != ""]
            return ", ".join(keys) if keys else pd.NA

        text = str(raw)
        if text == "" or text.lower() in NULL_LIKE_VALUES:
            return pd.NA
        if not (text.startswith("{") and text.endswith("}")):
            return text

        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return text

        if isinstance(parsed, dict):
            keys = [str(key) for key in parsed.keys() if str(key) != ""]
            return ", ".join(keys) if keys else pd.NA
        return text

    out["keywords"] = out["keywords"].apply(parse_keyword_value).astype("string")
    parsed_mask = original_keywords.ne(out["keywords"].astype("string"))
    parsed_keyword_rows = out.loc[parsed_mask].copy().reset_index(drop=True)
    return out, parsed_keyword_rows


def build_clean_news(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_publication_date = out["publication_date"].astype("string").str.strip()
    raw_publication_date = raw_publication_date.str.replace(r"\s*\(GMT[+-]\d+\)\s*$", "", regex=True)
    publication_date_digits = raw_publication_date.fillna("").str.replace(r"\D", "", regex=True)
    zero_publication_date_mask = (
        publication_date_digits.ne("")
        & publication_date_digits.str.fullmatch(r"0+", na=False)
    )
    invalid_publication_date_mask = (
        raw_publication_date.isna()
        | raw_publication_date.fillna("").str.lower().isin(NULL_LIKE_VALUES)
        | zero_publication_date_mask
    )
    out["publication_date"] = parse_publication_date_series(out["publication_date"])
    out = out.loc[~invalid_publication_date_mask & out["publication_date"].notna()].copy()
    year_before_2010_rows = build_year_before_2010_rows(out)
    out = out.drop(index=year_before_2010_rows.index).copy()
    out = add_description_word_count(out)
    out = out.loc[out["description_word_count"].ge(5)].copy()
    out = out.drop_duplicates(subset=["link", "publication_date"], keep="first").copy()
    out = out.sort_values(["publication_date", "link"], kind="mergesort").reset_index(drop=True)

    _, normalize_domain = normalize_domain_column(out)
    normalized_keywords, parsed_keyword_rows = parse_dict_like_keywords(out)

    clean_news = pd.DataFrame(
        {
            "publication_date": out["publication_date"],
            "year": out["year"],
            "domain_norm": normalize_domain["domain"].astype("string"),
            "category": out["category"],
            "title": out["title"],
            "description": out["description"],
            "keywords_norm": normalized_keywords["keywords"].astype("string"),
        }
    )
    return clean_news.reset_index(drop=True)


def main() -> None:
    raw_df = load_input_data(INPUT_PATH)
    normalized_df = normalize_text_columns(raw_df, TEXT_COLUMNS)
    input_rows = len(raw_df)
    clean_news_df = build_clean_news(normalized_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_news_df.to_parquet(OUTPUT_PATH, index=False)

    print("Input path:", INPUT_PATH)
    print("Output path:", OUTPUT_PATH)
    print("Input rows:", input_rows)
    print("Clean news rows:", len(clean_news_df))

if __name__ == "__main__":
    main()
