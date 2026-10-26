from pathlib import Path
import re
import pandas as pd
from underthesea import sent_tokenize, word_tokenize

INPUT_PATH = Path("data/equity_news_clean_content.parquet")
OUTPUT_PATH = Path("data/equity_news_tokenized.parquet")

TEXT_COLUMN =  "content" 
TOKENIZED_COLUMN = "Tokenize_content"
TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"
TOTAL_TOKENIZER_COLUMN = "total_tokenizer"

TOKEN_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)
SPECIAL_CHARACTER_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)


def clean_token(token: str) -> str | None:
    token = SPECIAL_CHARACTER_PATTERN.sub(" ", token.strip().lower())
    token = WHITESPACE_PATTERN.sub("_", token).strip("_")
    if len(token) <= 1:
        return None
    if not TOKEN_PATTERN.search(token):
        return None
    if token[0].isdigit():
        return None
    return token


def tokenize_vietnamese_text(text: object) -> list[str]:
    tokenized_sentences = tokenize_vietnamese_text_by_sentence(text)
    return [token for sentence in tokenized_sentences for token in sentence]


def tokenize_vietnamese_text_by_sentence(text: object) -> list[list[str]]:
    if pd.isna(text):
        return []

    tokenized_sentences: list[list[str]] = []
    for sentence in sent_tokenize(str(text)):
        tokens = word_tokenize(sentence, format=None)
        cleaned_tokens = [clean_token(token) for token in tokens]
        cleaned_tokens = [token for token in cleaned_tokens if token is not None]

        if cleaned_tokens:
            tokenized_sentences.append(cleaned_tokens)

    return tokenized_sentences


def load_and_tokenize_equity_news(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = df.copy()
    out[TOKENIZED_SENTENCES_COLUMN] = out[TEXT_COLUMN].apply(tokenize_vietnamese_text_by_sentence)
    out[TOKENIZED_COLUMN] = out[TOKENIZED_SENTENCES_COLUMN].apply(
        lambda sentences: [token for sentence in sentences for token in sentence]
    )
    out[TOTAL_TOKENIZER_COLUMN] = out[TOKENIZED_COLUMN].apply(len)
    return out


def main() -> None:
    df = load_and_tokenize_equity_news(INPUT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved tokenized dataframe to: {OUTPUT_PATH}")
    print("Rows:", len(df))
    print("Total sentences:", df[TOKENIZED_SENTENCES_COLUMN].apply(len).sum())
    print("Total tokens:", df[TOTAL_TOKENIZER_COLUMN].sum())


if __name__ == "__main__":
    main()
