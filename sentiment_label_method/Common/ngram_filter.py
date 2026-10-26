from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import NGRAM_SEPARATOR
from News.Build_sentiment_label.Common.stopword_utils import has_stopword_boundary
from News.Build_sentiment_label.Common.tf_df_utils import (
    add_candidate_statistics,
    infer_total_documents,
    split_by_mask,
    validate_ngram_terms,
)


def scaled_min_df_by_ngram(
    total_documents: int,
    min_df_ratio_by_ngram: dict[int, float],
    floor: int = 2,
) -> dict[int, int]:
    return {
        ngram_n: max(floor, math.ceil(total_documents * ratio))
        for ngram_n, ratio in min_df_ratio_by_ngram.items()
    }


def build_min_df_mask(
    term_stats: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
) -> pd.Series:
    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    return term_stats["df"].lt(min_df_threshold)


def choose_ngram_terms(
    ngram_terms_df: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
    max_df_ratio: float,
    total_documents: int | None = None,
    remove_stopwords: bool = True,
    stopwords: set[str] | None = None,
    return_groups: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    validate_ngram_terms(ngram_terms_df)
    total_documents = (
        infer_total_documents(ngram_terms_df)
        if total_documents is None
        else total_documents
    )
    term_stats = add_candidate_statistics(
        term_df=ngram_terms_df,
        total_documents=total_documents,
    )
    term_stats["rejection_reason"] = None

    if remove_stopwords:
        stopwords = stopwords or set()
        stopword_boundary_mask = term_stats["term"].apply(
            lambda term: has_stopword_boundary(
                term,
                stopwords=stopwords,
                separator=NGRAM_SEPARATOR,
            )
        )
    else:
        stopword_boundary_mask = pd.Series(False, index=term_stats.index)

    non_stopword_df, stopword_boundary_df = split_by_mask(
        term_stats,
        stopword_boundary_mask,
    )
    stopword_boundary_df["rejection_reason"] = "stopword_boundary"

    below_min_df_mask = build_min_df_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
    )
    min_df_pass_df, below_min_df_df = split_by_mask(
        non_stopword_df,
        below_min_df_mask,
    )
    below_min_df_df["rejection_reason"] = "below_min_df_by_ngram"

    above_max_df_ratio_mask = min_df_pass_df["df_ratio"].gt(max_df_ratio)
    candidate_terms_df, above_max_df_ratio_df = split_by_mask(
        min_df_pass_df,
        above_max_df_ratio_mask,
    )
    above_max_df_ratio_df["rejection_reason"] = "above_max_df_ratio"

    candidate_terms_df = candidate_terms_df.drop(columns=["rejection_reason"])
    candidate_terms_df = candidate_terms_df.sort_values(
        by=["candidate_score", "tf", "df", "ngram_n", "term"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    if not return_groups:
        return candidate_terms_df

    return {
        "all_ngram_terms_df": term_stats.reset_index(drop=True),
        "stopword_boundary_df": stopword_boundary_df,
        "below_min_df_df": below_min_df_df,
        "above_max_df_ratio_df": above_max_df_ratio_df,
        "candidate_terms_df": candidate_terms_df,
    }
