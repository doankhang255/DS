from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

INPUT_PATH = Path("data/equity_news.parquet")
OUTPUT_PATH = Path("data/equity_news_clean_content.parquet")

CONTENT_REMOVE_PHRASES = ("Khóa học online - Phân tích Ngành",)
CONTENT_KEEP_START = ("DIC Corp hoàn tất mua thêm để nâng sở hữu công ty con DIC Hospitality lên 99,36% vốn")
CONTENT_KEEP_END = ("Chủ tịch Nam Việt đăng ký thoái toàn bộ 900.000 cổ phiếu -")
RELATED_CONTENT_MARKER = (
    "Có thể bạn quan tâm",
    "Xem đáp án tại đây",
    "Xem thêm tại đây",
)


def remove_records_by_content1(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:

    content_series = df["content"].astype("string")
    remove_mask = pd.Series(False, index=df.index)
    for phrase in CONTENT_REMOVE_PHRASES:
        remove_mask = remove_mask | content_series.str.contains(
            phrase,
            case=False,
            na=False,
            regex=False,
        )

    removed_rows = int(remove_mask.sum())
    out = df.loc[~remove_mask].copy()
    print("Rows after content filter:", len(out))
    return out, removed_rows


def keep_content_between_dic_and_nav(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    def keep_between_markers(raw: object) -> tuple[object, bool]:
        if pd.isna(raw):
            return raw, False

        text = str(raw)
        search_text = text.casefold()
        start_marker = CONTENT_KEEP_START.casefold()
        end_marker = CONTENT_KEEP_END.casefold()

        start_index = search_text.find(start_marker)
        if start_index == -1:
            return raw, False

        content_start_index = start_index + len(CONTENT_KEEP_START)
        end_index = search_text.find(end_marker, content_start_index)
        if end_index == -1:
            return raw, False

        return text[content_start_index:end_index].strip(), True

    out = df.copy()
    cleaned_values = []
    changed_rows = 0
    for value in out["content"]:
        cleaned_value, changed = keep_between_markers(value)
        cleaned_values.append(cleaned_value)
        changed_rows += int(changed)

    out["content"] = cleaned_values
    print("Rows cleaned by DIC/Nav content range:", changed_rows)
    return out, changed_rows


def remove_title_description_from_content(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required_columns = {"content", "title", "description"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input data must contain columns: {sorted(missing_columns)}")

    def build_flexible_pattern(value: object) -> str | None:
        if pd.isna(value):
            return None

        text = str(value).strip()
        if text == "":
            return None

        parts = re.split(r"\s+", text)
        return r"\s+".join(re.escape(part) for part in parts)

    def clean_row_content(row: pd.Series) -> tuple[object, bool]:
        if pd.isna(row["content"]):
            return row["content"], False

        content = str(row["content"])
        cleaned_content = content
        patterns = [
            pattern
            for pattern in (
                build_flexible_pattern(row["description"]),
                build_flexible_pattern(row["title"]),
            )
            if pattern is not None
        ]
        patterns = sorted(set(patterns), key=len, reverse=True)

        for pattern in patterns:
            cleaned_content = re.sub(
                pattern,
                " ",
                cleaned_content,
                flags=re.IGNORECASE,
            )

        cleaned_content = re.sub(r"[ \t]+", " ", cleaned_content)
        cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)
        cleaned_content = cleaned_content.strip()
        return cleaned_content, cleaned_content != content

    out = df.copy()
    cleaned_values = []
    changed_rows = 0
    for _, row in out.iterrows():
        cleaned_value, changed = clean_row_content(row)
        cleaned_values.append(cleaned_value)
        changed_rows += int(changed)

    out["content"] = cleaned_values
    print("Rows cleaned by title/description removal:", changed_rows)
    return out, changed_rows


def keep_content_before_interest_marker(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:

    def keep_before_marker(raw: object) -> tuple[object, bool]:
        if pd.isna(raw):
            return raw, False

        text = str(raw)
        marker_indexes = [
            marker_index
            for marker in RELATED_CONTENT_MARKER
            if (marker_index := text.casefold().find(marker.casefold())) != -1
        ]
        if not marker_indexes:
            return raw, False

        marker_index = min(marker_indexes)
        return text[:marker_index].strip(), True

    out = df.copy()
    cleaned_values = []
    changed_rows = 0
    for value in out["content"]:
        cleaned_value, changed = keep_before_marker(value)
        cleaned_values.append(cleaned_value)
        changed_rows += int(changed)

    out["content"] = cleaned_values
    print("Rows cleaned by related-content marker:", changed_rows)
    return out, changed_rows


def replace_short_content_with_description(
    df: pd.DataFrame,
    min_word_count: int = 30,
) -> tuple[pd.DataFrame, int]:
    required_columns = {"content", "description"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input data must contain columns: {sorted(missing_columns)}")

    def count_words(raw: object) -> int:
        if pd.isna(raw):
            return 0

        text = str(raw).strip()
        if text == "":
            return 0

        normalized_text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return len(normalized_text.split())

    def clean_description(raw: object) -> object:
        if pd.isna(raw):
            return raw

        description = str(raw).strip()
        marker_indexes = [
            marker_index
            for marker in RELATED_CONTENT_MARKER
            if (marker_index := description.casefold().find(marker.casefold())) != -1
        ]
        if marker_indexes:
            description = description[: min(marker_indexes)]

        return description.strip()

    out = df.copy()
    content_word_count = out["content"].apply(count_words)
    short_content_mask = content_word_count.lt(min_word_count)
    out.loc[short_content_mask, "content"] = out.loc[
        short_content_mask,
        "description",
    ].apply(clean_description)

    replaced_rows = int(short_content_mask.sum())
    print("Rows replaced with description because content < 30 words:", replaced_rows)
    return out, replaced_rows


def main() -> None:
    equity_news_df = pd.read_parquet(INPUT_PATH)
    print("Input rows:", len(equity_news_df))

    content_clean_df, removed_by_content_rows = remove_records_by_content1(equity_news_df)
    content_clean_df, cleaned_range_rows = keep_content_between_dic_and_nav(content_clean_df)
    content_clean_df, cleaned_related_marker_rows = keep_content_before_interest_marker(content_clean_df)
    content_clean_df, cleaned_title_description_rows = remove_title_description_from_content(content_clean_df)
    content_clean_df, replaced_short_content_rows = replace_short_content_with_description(content_clean_df)

    # Sắp theo ngày đăng (trùng giờ thì giữ thứ tự gốc); source_row_id = số thứ tự dòng sau khi
    # sắp, là khóa nối với nhãn GT và các file article_scores
    content_clean_df = content_clean_df.sort_values("publication_date", kind="mergesort").reset_index(drop=True)
    content_clean_df.insert(0, "source_row_id", range(len(content_clean_df)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content_clean_df.to_parquet(OUTPUT_PATH, index=False)

    print("Output path:", OUTPUT_PATH)
    print("Rows removed by content filter:", removed_by_content_rows)
    print("Rows cleaned by DIC/Nav content range:", cleaned_range_rows)
    print("Rows cleaned by related-content marker:", cleaned_related_marker_rows)
    print("Rows cleaned by title/description removal:", cleaned_title_description_rows)
    print("Rows replaced with description because content < 30 words:", replaced_short_content_rows)
    print("Content clean rows:", len(content_clean_df))


if __name__ == "__main__":
    main()
