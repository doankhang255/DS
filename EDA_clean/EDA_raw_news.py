from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NEWS_DIR = PROJECT_ROOT / "data_news"
INPUT_PATH = DATA_NEWS_DIR / "dataset_news.parquet"

TEXT_COLUMNS = [
    "link",
    "domain",
    "category",
    "title",
    "description",
    "keywords",
    "author",
]
NULL_LIKE_VALUES = {"", "0", "null", "nan", "none"}


def load_input_data(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")
    elif source_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        raise ValueError("INPUT_PATH must point to a .parquet file.")
    return df


def normalize_text_columns(df: pd.DataFrame, text_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in text_columns:
        out[column] = out[column].astype("string")
        out[column] = out[column].str.replace(r"[\x00-\x1F\x7F-\x9F]", " ", regex=True)
        out[column] = out[column].str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def build_missing_records(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in df.columns:
        series = df[column]
        if column == "description":
            missing_mask = series.apply(count_words_in_text).eq(0)
        elif pd.api.types.is_string_dtype(series) or str(series.dtype) in {"object", "string"}:
            text_series = series.astype("string").str.strip()
            missing_value_mask = text_series.isna()
            null_like_mask = text_series.fillna("").str.lower().isin(NULL_LIKE_VALUES)
            missing_mask = missing_value_mask | null_like_mask

            if column == "publication_date":
                publication_date_text = text_series.fillna("")
                publication_date_text = publication_date_text.str.replace(r"\s*\(GMT[+-]\d+\)\s*$", "", regex=True)
                publication_date_digits = publication_date_text.str.replace(r"\D", "", regex=True)
                zero_publication_date_mask = (
                    publication_date_digits.ne("")
                    & publication_date_digits.str.fullmatch(r"0+", na=False)
                )
                missing_mask = missing_mask | zero_publication_date_mask
        else:
            missing_mask = series.isna()

        records.append(
            {
                "column": column,
                "number_of_missing_rows": int(missing_mask.sum()),
            }
        )
    missing_records = pd.DataFrame(records)
    return missing_records


def parse_publication_date_series(series: pd.Series) -> pd.Series:
    text_series = series.astype("string").str.strip()
    text_series = text_series.str.replace(r"\s*\(GMT[+-]\d+\)\s*$", "", regex=True)
    text_series = text_series.str.replace(r"([+-]\d{2}:?\d{2}|Z)$", "", regex=True)
    text_series = text_series.mask(text_series.str.lower().isin(NULL_LIKE_VALUES))
    text_series = text_series.mask(text_series.eq("0000-00-00 00:00:00"))
    # Ngày gốc dạng năm-tháng-ngày ("2010-08-02 14:57:00.0"): đọc theo ISO trước. Không dùng
    # dayfirst=True cho chuỗi ISO - pandas sẽ đọc 2010-08-02 thành ngày 8 tháng 2 (lỗi đảo
    # ngày/tháng với mọi ngày <= 12, đã sửa 2026-10-20). Chỉ chuỗi không phải ISO mới đọc ngày trước.
    parsed_publication_date = pd.to_datetime(text_series, errors="coerce", format="ISO8601")
    rest = parsed_publication_date.isna() & text_series.notna()
    if rest.any():
        parsed_publication_date[rest] = pd.to_datetime(
            text_series[rest],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )
    return parsed_publication_date


def count_words_in_text(raw: object) -> int:
    if pd.isna(raw):
        return 0

    text = str(raw).strip()
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text) 

    if text == "" or text.lower() in NULL_LIKE_VALUES:
        return 0

    normalized_text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = normalized_text.split()
    word_tokens = [
        token
        for token in tokens
        if token.lower() not in NULL_LIKE_VALUES
        and re.search(r"[^\W\d_]", token, flags=re.UNICODE)
    ]
    return len(word_tokens)


def add_description_word_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["description_word_count"] = out["description"].apply(count_words_in_text)
    return out


def calculate_description_word_count_percentiles(df: pd.DataFrame) -> dict[str, float]:
    series = df["description_word_count"].dropna()
    if series.empty:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "p5": 0.0}

    return {
        "p50": round(float(series.quantile(0.50)), 2),
        "p90": round(float(series.quantile(0.90)), 2),
        "p99": round(float(series.quantile(0.99)), 2),
        "p5": round(float(series.quantile(0.05)), 2)
    }


def split_missing_description_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = add_description_word_count(df)
    has_description_mask = out["description_word_count"].gt(0)
    has_description_df = out.loc[has_description_mask].copy().reset_index(drop=True)
    missing_description_df = out.loc[~has_description_mask].copy().reset_index(drop=True)
    return has_description_df, missing_description_df


def build_duplicate_link_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df.duplicated(subset=["link", "publication_date"], keep=False)]
        .sort_values(["link", "publication_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_year_record_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["publication_date"] = parse_publication_date_series(out["publication_date"])
    out["year_from_publication_date"] = out["publication_date"].dt.year.astype("Int64")
    out["analysis_year"] = out["year_from_publication_date"].where(
        out["year_from_publication_date"].notna(),
        out["year"],
    )
    year_record_counts = (
        out.loc[out["analysis_year"].notna()]
        .groupby("analysis_year", dropna=False)
        .size()
        .reset_index(name="number_of_records")
        .rename(columns={"analysis_year": "year"})
        .sort_values("year", kind="mergesort")
        .reset_index(drop=True)
    )
    return year_record_counts


def build_year_before_2010_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["publication_date"] = parse_publication_date_series(out["publication_date"])
    out["year_from_publication_date"] = out["publication_date"].dt.year.astype("Int64")
    analysis_year = out["year_from_publication_date"].where(
        out["year_from_publication_date"].notna(),
        out["year"],
    )
    valid_analysis_year_mask = analysis_year.notna() & analysis_year.ge(1901)
    year_before_2010_rows = out.loc[valid_analysis_year_mask & analysis_year.lt(2010)].copy()
    return year_before_2010_rows


def build_unique_value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = df[column]
    if pd.api.types.is_string_dtype(series) or str(series.dtype) in {"object", "string"}:
        text_series = series.astype("string").str.strip()
        valid_mask = ~(text_series.isna() | text_series.fillna("").str.lower().isin(NULL_LIKE_VALUES))
        value_series = text_series.loc[valid_mask]
    else:
        valid_mask = series.notna()
        value_series = series.loc[valid_mask]

    summary = (
        value_series
        .astype("string")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="row_count")
        .sort_values(["row_count", column], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def build_unique_domains(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = out["domain"].astype("string").str.strip().str.lower()
    unique_domains = build_unique_value_counts(out, "domain")[["domain"]].copy()
    return unique_domains


def build_unique_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["category"] = out["category"].astype("string").str.strip().str.lower()
    unique_categories = build_unique_value_counts(out, "category")[["category"]].copy()
    return unique_categories


def build_domain_record_counts(df: pd.DataFrame) -> pd.DataFrame:
    domain_series = df["domain"].astype("string").str.strip().str.lower()
    valid_domain_mask = ~(domain_series.isna() | domain_series.fillna("").isin(NULL_LIKE_VALUES))
    domain_record_counts = (
        domain_series.loc[valid_domain_mask]
        .value_counts(dropna=False)
        .rename_axis("domain")
        .reset_index(name="number_of_records")
        .sort_values(["number_of_records", "domain"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return domain_record_counts


def main() -> None:
    raw_df = load_input_data(INPUT_PATH)
    normal_text = normalize_text_columns(raw_df, TEXT_COLUMNS)
    parsed_publication_date = parse_publication_date_series(normal_text["publication_date"])
    has_description_df, missing_description_df = split_missing_description_rows(normal_text)
    missing_records = build_missing_records(normal_text)
    duplicate_rows = build_duplicate_link_date_rows(normal_text)
    year_record_counts = build_year_record_counts(normal_text)
    unique_domains = build_unique_domains(normal_text)
    description_word_count_percentiles = calculate_description_word_count_percentiles(has_description_df)
    build_domain_record_counts_df = build_domain_record_counts(normal_text)
    unique_categories = build_unique_categories(normal_text)
    year_before_2010_rows = build_year_before_2010_rows(normal_text)


    print("Input records:", len(raw_df))
    print("Unique domains:", len(unique_domains))
    print("Duplicate rows by link + publication_date:", len(duplicate_rows))
    print("Unique categories:", len(unique_categories))
    if parsed_publication_date.notna().any():
        print("Min publication date:", parsed_publication_date.min().strftime("%Y-%m-%d %H:%M:%S.0"))
        print("Max publication date:", parsed_publication_date.max().strftime("%Y-%m-%d %H:%M:%S.0"))
    else:
        print("Min publication date:", pd.NA)
        print("Max publication date:", pd.NA)
    print("Rows with year < 2010:", len(year_before_2010_rows))
    print(f"Maximum description word count: {has_description_df['description_word_count'].max()}")
    print(f"Minimum description word count: {has_description_df['description_word_count'].min()}")
    print(f"Average description word count: {has_description_df['description_word_count'].mean():.2f}")
    print(f"Description word count p50: {description_word_count_percentiles['p50']}")
    print(f"Description word count p90: {description_word_count_percentiles['p90']}")
    print(f"Description word count p99: {description_word_count_percentiles['p99']}")
    print(f"Description word count p5: {description_word_count_percentiles['p5']}")
    
    for _, row in year_record_counts.iterrows():
        print(f"number of records in year {int(row['year'])}: {row['number_of_records']}")

    for _, row in missing_records.iterrows():
        print(f"number of missing rows in {row['column']}: {row['number_of_missing_rows']}")

    for _, row in build_domain_record_counts_df.iterrows():
        print(f"number of records per domain: {row['domain']} - {row['number_of_records']}")

if __name__ == "__main__":
    main()
