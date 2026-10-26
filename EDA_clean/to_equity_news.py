from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/clean_news_tokenized.parquet")
CATEGORIES_PATH = Path("data/categories.csv")
DOMAIN_NORM_PATH = Path("data/domain_norm.csv")
OUTPUT_PATH = Path("data/equity_news.parquet")


def normalize_for_match(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"[\x00-\x1F\x7F-\x9F]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def load_allowed_values(path: Path, column: str) -> set[str]:
    allowed_df = pd.read_csv(path)
    if column not in allowed_df.columns:
        raise ValueError(f"{path} must contain column: {column}")

    values = normalize_for_match(allowed_df[column]).dropna()
    values = values.loc[values.ne("")]
    return set(values)


def build_equity_news(df: pd.DataFrame) -> pd.DataFrame:
    category_values = load_allowed_values(CATEGORIES_PATH, "category")
    domain_values = load_allowed_values(DOMAIN_NORM_PATH, "domain_norm")

    out = df.copy()

    category_mask = normalize_for_match(out["category"]).isin(category_values)
    out = out.loc[category_mask].copy()
    print("Rows after category filter:", len(out))

    domain_mask = normalize_for_match(out["domain_norm"]).isin(domain_values)
    out = out.loc[domain_mask].copy()
    print("Rows after domain_norm filter:", len(out))

    return out.reset_index(drop=True)


def main() -> None:
    clean_news_df = pd.read_parquet(INPUT_PATH)
    print("Input rows:", len(clean_news_df))

    equity_news_df = build_equity_news(clean_news_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    equity_news_df.to_parquet(OUTPUT_PATH, index=False)

    print("Output path:", OUTPUT_PATH)
    print("Equity news rows:", len(equity_news_df))


if __name__ == "__main__":
    main()
