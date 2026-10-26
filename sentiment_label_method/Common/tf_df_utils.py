from __future__ import annotations

import math

import pandas as pd


REQUIRED_NGRAM_COLUMNS = {"term", "ngram_n", "tf", "df"}


def split_by_mask(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    failed_df = df.loc[mask].copy().reset_index(drop=True)
    passed_df = df.loc[~mask].copy().reset_index(drop=True)
    return passed_df, failed_df


def infer_total_documents(term_df: pd.DataFrame) -> int:
    if term_df.empty:
        return 0
    if "df_ratio" not in term_df.columns:
        raise ValueError("total_documents is required when df_ratio column is missing")

    valid_df_ratio = term_df["df_ratio"].gt(0)
    if not valid_df_ratio.any():
        return 0

    estimates = term_df.loc[valid_df_ratio, "df"] / term_df.loc[valid_df_ratio, "df_ratio"]
    return int(round(estimates.median()))


def validate_ngram_terms(term_df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_NGRAM_COLUMNS.difference(term_df.columns)
    if missing_columns:
        raise ValueError(f"Input n-gram terms must contain columns: {sorted(missing_columns)}")


def add_candidate_statistics(
    term_df: pd.DataFrame,
    total_documents: int,
) -> pd.DataFrame:
    out = term_df.copy()
    if "df_ratio" not in out.columns:
        out["df_ratio"] = out["df"] / total_documents if total_documents else 0.0

    out["avg_tf_per_doc"] = out["tf"] / out["df"]
    out["candidate_score"] = out["tf"] * out["df"].apply(
        lambda df: math.log((total_documents + 1) / (df + 1))
    )
    return out
