from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent

DATA_PATH = WORKSPACE_DIR / "非匿名版" / "outputs" / "data_new.csv"
LABEL_PATH = BASE_DIR / "label.csv"
LABEL_NEW_PATH = BASE_DIR / "label_new.csv"
DCARD_NAME_PATH = WORKSPACE_DIR / "非匿名版" / "outputs" / "dcard_name.csv"
SENTIMENT_PATH = BASE_DIR / "sentiment_results_upgraded.csv"
TRAINING_OUTPUT_PATH = BASE_DIR / "training_data_bert128.csv"
BERT_MODEL_NAME = "ckiplab/bert-base-chinese"
BERT_BATCH_SIZE = 16
BERT_ARTICLE_MAX_LEN = 512
BERT_POOLING = "mean_cls_over_chunks"
BERT_FIXED_COMPONENTS = 128
BERT_TEXT_CLEANING_VERSION = "remove_urls_keep_cjk_en_digit_punct_v1"
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


CATEGORY_KEYWORDS = {
    "cta_click": [
        "點我下單", "輸入代碼", "傳送門",
        "官方網站", "粉絲專頁", "粉專",
        "點入官網", "直接點這裡", "詳細資訊請見",
        "私訊", "留言", "加好友", "追蹤",
        "分享", "收藏", "報名", "預約",
        "下載", "填寫", "按讚",
    ],
    "promotion": [
        "讀者優惠", "折扣", "專屬優惠",
        "優惠碼", "限時團購", "開團",
        "團購價", "買一送一", "免費",
        "免運", "划算", "cp值", "省錢",
        "便宜", "下殺", "滿額", "福利",
        "附贈", "折價券", "優惠券",
    ],
    "urgency": [
        "手刀", "值得", "值回票價", "佛心",
        "誠意滿滿", "一定要試試", "必買",
        "必去", "必吃", "推推", "大推",
        "超推", "首選", "不來可惜",
        "趁現在", "回購", "必囤", "囤貨",
        "種草", "激推", "強推",
    ],
    "disclosure": [
        "感謝邀請", "合作邀約", "體驗文",
        "商業合作", "產品贊助", "sponsored",
        "ad", "pr", "愛體驗",
        "試用分享", "活動體驗",
    ],
    "platform_brand": [
        "kkday", "klook", "客路",
        "agoda", "booking.com", "travago",
        "tripadvisor", "網卡", "wifi機",
        "機加酒", "旅遊平安險", "新安東京",
        "富邦", "雄獅旅遊", "可樂旅遊",
        "東南旅遊", "五福旅遊", "易遊網",
        "蝦皮", "momo", "淘寶", "官網",
    ],
    "recommendation": [
        "真心推薦", "私心推薦", "真心",
    ],
    "lottery": [
        "抽獎", "文末抽獎", "留言抽獎",
    ],
}


DROP_COLUMNS = [
    "id",
    "articleId",
    "id_data",
    "articleId_from_data_order",
    "file_name",
    "articleId_from_file",
    "title",
    "content",
    "authorName",
    "authorSubtitle",
    "createdAt",
    "標註1",
    "標註2",
    "forumName",
    "forumAlias",
    "authorUseNickname",
    "authorUseNickname_data",
    "commentCount",
    "totalCommentCount",
    "linksCount",
    "edited",
    "likeCount",
    "collectionCount",
    "shareCount",
    "withImages",
    "withVideos",
    "imageCount",
    "videoCount",
    "authorHasCreatorBadge",
    "authorHasOfficialCreatorBadge",
    "CountEmoji",
    "chunks_count",
]


RENAME_COLUMNS = {
    "最終判定": "label",
    "edited_data": "edited",
    "likeCount_data": "likeCount",
    "collectionCount_data": "collectionCount",
    "shareCount_data": "shareCount",
    "commentCount_data": "commentCount",
    "totalCommentCount_data": "totalCommentCount",
    "linksCount_data": "linksCount",
    "withImages_data": "withImages",
    "withVideos_data": "withVideos",
    "imageCount_data": "imageCount",
    "videoCount_data": "videoCount",
    "authorHasCreatorBadge_data": "authorHasCreatorBadge",
    "authorHasOfficialCreatorBadge_data": "authorHasOfficialCreatorBadge",
}


def resolve_data_path() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    raise FileNotFoundError(f"Cannot find data.csv: {DATA_PATH}")


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_positive_label_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["1", "1.0", "true", "t", "yes", "y"])


def load_label_new_positive_rows(existing_ids: pd.Series) -> pd.DataFrame:
    if not LABEL_NEW_PATH.exists():
        print(f"label_new.csv not found, skipping new positive labels: {LABEL_NEW_PATH}")
        return pd.DataFrame()

    label_new = pd.read_csv(
        LABEL_NEW_PATH,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    required_columns = {"id", "human_labeling"}
    missing_columns = sorted(required_columns - set(label_new.columns))
    if missing_columns:
        raise KeyError(f"label_new.csv missing required columns: {missing_columns}")

    positive_rows = label_new.loc[
        normalize_positive_label_mask(label_new["human_labeling"])
    ].copy()

    existing_id_keys = set(normalize_key(existing_ids).dropna())
    positive_rows["_merge_id_key"] = normalize_key(positive_rows["id"])
    duplicate_existing_count = positive_rows["_merge_id_key"].isin(existing_id_keys).sum()
    positive_rows = positive_rows.loc[
        ~positive_rows["_merge_id_key"].isin(existing_id_keys)
    ].copy()

    if DCARD_NAME_PATH.exists():
        dcard_name = pd.read_csv(DCARD_NAME_PATH, encoding="utf-8-sig")
        dcard_name = dcard_name.drop_duplicates(subset="id")
        positive_rows = positive_rows.merge(
            dcard_name[["id", "articleId", "title", "content"]],
            on="id",
            how="left",
            suffixes=("", "_dcard"),
            validate="many_to_one",
        )
        if "articleId_dcard" in positive_rows.columns:
            positive_rows["articleId"] = positive_rows["articleId"].combine_first(
                positive_rows["articleId_dcard"]
            )
    else:
        positive_rows = positive_rows.rename(
            columns={
                "title_x": "title",
                "content_x": "content",
            }
        )

    for fallback_column, target_column in (("title_x", "title"), ("content_x", "content")):
        if fallback_column in positive_rows.columns:
            if target_column not in positive_rows.columns:
                positive_rows[target_column] = positive_rows[fallback_column]
            else:
                positive_rows[target_column] = positive_rows[target_column].combine_first(
                    positive_rows[fallback_column]
                )

    positive_rows["最終判定"] = "T"
    positive_rows = positive_rows[["id", "articleId", "title", "content", "最終判定"]]
    positive_rows = positive_rows.drop_duplicates(subset="id", keep="first")

    print(f"label_new.csv rows: {len(label_new)}")
    print(f"label_new.csv human-labeled positive rows: {normalize_positive_label_mask(label_new['human_labeling']).sum()}")
    print(f"label_new positives already in label.csv skipped: {duplicate_existing_count}")
    print(f"label_new positives added: {len(positive_rows)}")

    return positive_rows


def merge_csv_files() -> pd.DataFrame:
    data_path = resolve_data_path()
    data = pd.read_csv(data_path)
    labels = pd.read_csv(LABEL_PATH)
    sentiment = pd.read_csv(SENTIMENT_PATH)

    labels = labels.loc[:, ~labels.columns.str.startswith("Unnamed:")]
    new_positive_labels = load_label_new_positive_rows(labels["id"])
    if not new_positive_labels.empty:
        labels = pd.concat([labels, new_positive_labels], ignore_index=True, sort=False)

    article_col = next(
        (col for col in ("articleID", "articleId", "article_id") if col in labels.columns),
        None,
    )
    if article_col is None:
        raise KeyError("Cannot find articleID/articleId/article_id column in label.csv")

    data["_merge_id_key"] = normalize_key(data["id"])
    labels["_merge_id_key"] = normalize_key(labels["id"])
    labels["_merge_article_key"] = normalize_key(labels[article_col])
    data["_merge_occurrence_key"] = data.groupby("_merge_id_key").cumcount()
    labels["_merge_occurrence_key"] = labels.groupby("_merge_id_key").cumcount()

    sentiment["articleId_from_file"] = (
        sentiment["file_name"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.txt$", "", regex=True)
    )
    sentiment["_merge_article_key"] = normalize_key(sentiment["articleId_from_file"])

    merged = labels.merge(
        data,
        on=["_merge_id_key", "_merge_occurrence_key"],
        how="left",
        suffixes=("", "_data"),
        validate="one_to_one",
    )
    merged = merged.merge(
        sentiment,
        on="_merge_article_key",
        how="left",
        suffixes=("", "_sentiment"),
        validate="many_to_one",
    )

    data_matches = merged["id_data"].notna().sum() if "id_data" in merged else 0
    sentiment_matches = merged["file_name"].notna().sum()

    merged = merged.drop(columns=["_merge_id_key", "_merge_article_key", "_merge_occurrence_key"])

    print(f"data.csv path: {data_path}")
    print(f"data.csv rows: {len(data)}")
    print(f"label.csv rows: {len(labels)}")
    print(f"Matched data.csv rows: {data_matches}")
    print(f"Matched sentiment rows: {sentiment_matches}")

    return merged


def add_keyword_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    full_text = (df["title"].fillna("") + " " + df["content"].fillna("")).str.lower()
    word_counts = df["wordCount"].fillna(0).replace(0, 1)

    for category, keyword_list in CATEGORY_KEYWORDS.items():
        cleaned_keywords = list(set([keyword.lower() for keyword in keyword_list]))
        total_counts_for_category = np.zeros(len(df))

        for keyword in cleaned_keywords:
            total_counts_for_category += full_text.str.count(keyword).values

        df[f"cat_freq_{category}"] = total_counts_for_category / word_counts

    return df


def is_bert_allowed_character(char: str) -> bool:
    if char.isspace():
        return True
    if "\u4e00" <= char <= "\u9fff":
        return True
    if char.isascii() and char.isalnum():
        return True
    return unicodedata.category(char).startswith("P")


def clean_text_for_bert(text: str) -> str:
    text = URL_PATTERN.sub(" ", str(text))
    cleaned = "".join(char if is_bert_allowed_character(char) else " " for char in text)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_bert_chunks(text_list, max_len, tokenizer):
    from transformers.utils import logging as transformers_logging

    chunk_token_limit = max_len - tokenizer.num_special_tokens_to_add(pair=False)
    if chunk_token_limit <= 0:
        raise ValueError(f"max_len={max_len} is too small for tokenizer special tokens")
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Tokenizer must provide cls_token_id and sep_token_id for BERT chunks")

    chunk_records = []
    article_chunk_indices = []

    for article_index, text in enumerate(text_list):
        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            tokens = tokenizer.tokenize(str(text))
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if not token_ids:
                token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(""))
        finally:
            transformers_logging.set_verbosity(previous_verbosity)

        current_indices = []
        for start in range(0, max(len(token_ids), 1), chunk_token_limit):
            chunk_ids = token_ids[start : start + chunk_token_limit]
            input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
            current_indices.append(len(chunk_records))
            chunk_records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "article_index": article_index,
                }
            )

        article_chunk_indices.append(current_indices)

    return chunk_records, article_chunk_indices


def extract_bert_in_batches(text_list, max_len, desc_text, tokenizer, model, device):
    import torch
    from tqdm import tqdm

    chunk_records, article_chunk_indices = build_bert_chunks(text_list, max_len, tokenizer)
    hidden_size = model.config.hidden_size
    article_sums = np.zeros((len(text_list), hidden_size), dtype=np.float64)
    article_counts = np.zeros(len(text_list), dtype=np.int64)

    for start in tqdm(range(0, len(chunk_records), BERT_BATCH_SIZE), desc=desc_text):
        batch_records = chunk_records[start : start + BERT_BATCH_SIZE]
        batch = tokenizer.pad(
            {
                "input_ids": [record["input_ids"] for record in batch_records],
                "attention_mask": [record["attention_mask"] for record in batch_records],
            },
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for record, cls_vector in zip(batch_records, cls_vectors):
            article_index = record["article_index"]
            article_sums[article_index] += cls_vector
            article_counts[article_index] += 1

    if np.any(article_counts == 0):
        empty_indices = np.where(article_counts == 0)[0].tolist()
        raise RuntimeError(f"No BERT chunks generated for rows: {empty_indices}")

    print(f"{desc_text}: {len(chunk_records)} chunks for {len(text_list)} texts")
    return (article_sums / article_counts[:, None]).astype(np.float32)


def reduce_bert_embeddings(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    max_components = min(embeddings.shape)
    if max_components <= 1:
        print("Skipping BERT PCA because there are not enough samples/components")
        return embeddings.astype(np.float32)

    n_components = min(BERT_FIXED_COMPONENTS, max_components)
    pca = PCA(
        n_components=n_components,
        svd_solver="full",
        random_state=42,
    )
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = float(pca.explained_variance_ratio_.sum())

    print(
        "BERT article PCA fixed components: "
        f"{embeddings.shape[1]} -> {reduced_embeddings.shape[1]} features, "
        f"explained variance={explained_variance:.4f}"
    )
    if n_components < BERT_FIXED_COMPONENTS:
        print(
            "BERT PCA used fewer than 128 components because the sample/feature "
            f"limit is {max_components}"
        )

    return reduced_embeddings.astype(np.float32)


def add_bert_article_vector_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title_texts = df["title"].fillna("").astype(str).tolist()
    content_texts = df["content"].fillna("").astype(str).tolist()

    article_texts = [
        clean_text_for_bert(f"{title}\n\n{content}")
        for title, content in zip(title_texts, content_texts)
    ]

    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading BERT model: {BERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("Extracting BERT article CLS features")
    embeddings = extract_bert_in_batches(
        article_texts, BERT_ARTICLE_MAX_LEN, "BERT article", tokenizer, model, device
    )

    reduced_embeddings = reduce_bert_embeddings(embeddings)
    for index in range(reduced_embeddings.shape[1]):
        df[f"bert_article_pca_{index + 1:03d}"] = reduced_embeddings[:, index]

    return df


def clean_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    original_rows = len(df)

    df = add_bert_article_vector_features(df)
    df = add_keyword_frequency_features(df)
    cleaned = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
    cleaned = cleaned.rename(columns=RENAME_COLUMNS)

    if "label" not in cleaned.columns:
        raise KeyError("Cannot find 最終判定 column to create label")

    if "firstCommentTimeDiff" in cleaned.columns:
        max_value = cleaned["firstCommentTimeDiff"].max(skipna=True)
        fill_value = max_value * 2
        missing_count = cleaned["firstCommentTimeDiff"].isna().sum()
        cleaned["firstCommentTimeDiff"] = cleaned["firstCommentTimeDiff"].fillna(fill_value)
    else:
        fill_value = None
        missing_count = 0

    label = cleaned.pop("label").astype("category")
    cleaned.insert(0, "label", label)

    constant_cols = [
        col for col in cleaned.columns if col != "label" and cleaned[col].nunique(dropna=False) <= 1
    ]
    if constant_cols:
        cleaned = cleaned.drop(columns=constant_cols)

    rows_before_dropna = len(cleaned)
    cleaned = cleaned.dropna(axis=0).reset_index(drop=True)
    dropped_rows = rows_before_dropna - len(cleaned)

    cleaned.to_csv(TRAINING_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Training output: {TRAINING_OUTPUT_PATH}")
    print(f"Original rows: {original_rows}")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Rows dropped by dropna: {dropped_rows}")
    print(f"Original columns: {len(df.columns)}")
    print(f"Cleaned columns: {len(cleaned.columns)}")
    print(f"firstCommentTimeDiff missing filled: {missing_count}")
    if fill_value is not None:
        print(f"firstCommentTimeDiff fill value: {fill_value}")
    print("Label distribution:")
    print(cleaned["label"].value_counts(dropna=False).to_string())

    return cleaned, TRAINING_OUTPUT_PATH


def main() -> None:
    merged = merge_csv_files()
    cleaned, output_path = clean_training_data(merged)
    total_missing = int(cleaned.isna().sum().sum())
    print(f"Total missing values in {output_path.name}: {total_missing}")


if __name__ == "__main__":
    main()
