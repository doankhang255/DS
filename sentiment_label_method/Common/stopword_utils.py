from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCES_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "final_seed"
DEFAULT_STOPWORDS_PATH = RESOURCES_DIR / "vietnamese-stopwords-dash.txt"


def load_stopwords(path: Path = DEFAULT_STOPWORDS_PATH) -> set[str]:
    with open(path, encoding="utf-8") as file:
        return {line.strip().casefold() for line in file if line.strip()}


def has_stopword_boundary(
    term: str,
    stopwords: set[str],
    separator: str = " ",
) -> bool:
    if not stopwords:
        return False

    tokens = str(term).split(separator)
    if not tokens:
        return False

    return tokens[0].casefold() in stopwords or tokens[-1].casefold() in stopwords
