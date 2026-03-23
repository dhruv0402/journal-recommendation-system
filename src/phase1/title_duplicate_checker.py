from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Lightweight model for titles (fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

TITLE_DUP_THRESHOLD = 0.90
TITLE_NEAR_THRESHOLD = 0.75

_TITLE_EMBEDDINGS = None


def preload_title_embeddings(dataset_titles: list[str]):
    global _TITLE_EMBEDDINGS
    _TITLE_EMBEDDINGS = model.encode(
        dataset_titles,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False
    )


def check_title_against_dataset(user_title: str, dataset_titles: list[str]):
    global _TITLE_EMBEDDINGS

    if _TITLE_EMBEDDINGS is None:
        preload_title_embeddings(dataset_titles)

    user_emb = model.encode(user_title, normalize_embeddings=True)

    sims = cosine_similarity([user_emb], _TITLE_EMBEDDINGS)[0]
    max_sim = float(np.max(sims))

    if max_sim >= TITLE_DUP_THRESHOLD:
        verdict = "EXACT_MATCH"
    elif max_sim >= TITLE_NEAR_THRESHOLD:
        verdict = "NEAR_MATCH"
    else:
        verdict = "OK"

    return {
        "status": verdict,
        "confidence": round(max_sim, 3)
    }