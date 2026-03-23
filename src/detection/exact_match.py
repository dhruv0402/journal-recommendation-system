from src.detection.normalize import normalize_title


def exact_match(title: str, df):
    if not title:
        return []

    return df[df["article_title"] == title]["article_title"].tolist()


def normalized_match(title: str, df):
    if not title:
        return []

    normalized_input = normalize_title(title)

    df["_normalized"] = df["article_title"].apply(normalize_title)

    return df[df["_normalized"] == normalized_input]["article_title"].tolist()