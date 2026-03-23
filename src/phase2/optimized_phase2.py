import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.phase2.abstract_aggregation import aggregate_abstract_results
from src.phase2.final_decision import make_final_decision
from src.phase2.journal_heading_recommender import recommend_journal_headings
from src.phase2.scope_reranker import rerank_with_scope
from src.phase2.learning_reranker import rerank_with_learning

import os
import pickle

from src.rag.rag_engine import RAGEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
META_PATH = os.path.join(BASE_DIR, "data", "faiss_meta.pkl")

# ---------------- GLOBALS ----------------
_model = SentenceTransformer("all-MiniLM-L6-v2")
_index = None
_metadata = None
_rag_engine = None
_query_cache = {}


# ---------------- PRELOAD ----------------
def preload_phase2(df):
    global _index, _metadata

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("[Phase2] Loading FAISS index from disk...")
        _index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)

        print("[Phase2] FAISS loaded.")
    else:
        print("[Phase2] Building FAISS index...")

        abstracts = df["abstract"].fillna("").tolist()
        journals = df["journal_name"].tolist()

        embeddings = _model.encode(
            abstracts,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=True,
        )

        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(np.array(embeddings))

        _index = index
        _metadata = list(zip(abstracts, journals))

        os.makedirs("data", exist_ok=True)
        faiss.write_index(_index, INDEX_PATH)

        with open(META_PATH, "wb") as f:
            pickle.dump(_metadata, f)

        print("[Phase2] FAISS built and saved.")

    global _rag_engine
    if _rag_engine is None:
        try:
            print("[Phase2] Initializing RAG engine...")
            _rag_engine = RAGEngine(df)
        except Exception as e:
            print("[Phase2] RAG init failed:", e)


def run_phase2_fast(user_abstract: str, top_k: int = 50, training_mode: bool = False):
    """
    Fast Phase 2 using FAISS retrieval
    """

    global _index, _metadata

    if _index is None:
        raise Exception("Phase2 not initialized. Call preload_phase2() first")

    # -------- EMBEDDING CACHE --------
    if user_abstract in _query_cache:
        user_embedding = _query_cache[user_abstract]
    else:
        user_embedding = _model.encode([user_abstract], normalize_embeddings=True)
        _query_cache[user_abstract] = user_embedding

    # FAISS search (O(log N))
    k = min(top_k, 50)
    scores, indices = _index.search(user_embedding, k)

    article_results = []

    for score, idx in zip(scores[0], indices[0]):
        abstract, journal = _metadata[idx]

        article_results.append(
            {"journal_name": journal, "similarity": float(score), "abstract": abstract}
        )

    # ---------------- NORMALIZATION ----------------
    sims = [a["similarity"] for a in article_results]

    if sims:
        min_s, max_s = min(sims), max(sims)

        for a in article_results:
            if max_s > min_s:
                a["similarity"] = round((a["similarity"] - min_s) / (max_s - min_s), 3)
            else:
                a["similarity"] = 0.0

            a["similarity"] = max(0.0, min(a["similarity"], 1.0))

    # ---------------- AGGREGATION ----------------
    journal_predictions = aggregate_abstract_results(article_results)

    # -------- TRAINING MODE SHORT-CIRCUIT --------
    if training_mode:
        return {"journal_predictions": journal_predictions}

    # -------- APPLY RERANKERS --------
    journal_predictions = rerank_with_scope(user_embedding, journal_predictions)
    journal_predictions = rerank_with_learning(journal_predictions)

    raw_scores = [j.get("confidence", 0.0) for j in journal_predictions]

    # ---------------- UNCERTAINTY SIGNALS ----------------
    spread = 0.0

    if raw_scores:
        sorted_scores = sorted(raw_scores, reverse=True)
        top3 = sorted_scores[:3]

        if len(top3) >= 2:
            spread = top3[0] - top3[-1]

    # ---------------- KEEP RAW SCORES (NO NORMALIZATION) ----------------
    # Do NOT normalize reranker outputs — preserve ranking signal strength

    # ---------------- FINAL DECISION ----------------
    submission = make_final_decision(journal_predictions)

    # -------- SAFETY FIX: ensure dict --------
    if isinstance(submission, str):
        submission = {"journal": submission, "confidence": 0.0}

    # compute final confidence ONLY ONCE (no duplication)
    if raw_scores:
        sorted_scores = sorted(raw_scores, reverse=True)

        top1 = sorted_scores[0]
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        margin = max(0.0, top1 - top2)

        best_idx = int(np.argmax(raw_scores))
        best_similarity = journal_predictions[best_idx].get("similarity", 0.0)

        # improved weighting (entropy shouldn't kill score)
        confidence = 0.7 * best_similarity + 0.3 * (margin / (margin + 0.4))

        confidence = max(0.0, min(confidence, 1.0))
        confidence = round(confidence, 3)
    else:
        confidence = 0.0

    # -------- STABILITY FIX (avoid flip when close scores) --------
    if len(journal_predictions) > 1:
        sorted_preds = sorted(
            journal_predictions, key=lambda x: x.get("confidence", 0.0), reverse=True
        )
        top1 = sorted_preds[0].get("confidence", 0.0)
        top2 = sorted_preds[1].get("confidence", 0.0)
        if abs(top1 - top2) < 0.05:
            submission["journal"] = sorted_preds[0].get(
                "journal_name", submission.get("journal")
            )

    submission["confidence"] = confidence

    if confidence >= 0.65:
        final_decision = "Strong journal scope match"
    elif confidence >= 0.45:
        final_decision = "Partial journal scope match"
    else:
        final_decision = "Novel article – no strong journal scope match"

    # ---------------- HEADINGS ----------------
    recommended_headings = None
    if confidence >= 0.65:
        recommended_headings = recommend_journal_headings({}, top_k=3)

    best_journal = submission.get("journal", "No suitable journal")

    if confidence >= 0.65:
        final_recommendation = f"Submit to {best_journal}"
    elif confidence >= 0.45:
        final_recommendation = f"Possible fit: {best_journal}"
    else:
        final_recommendation = "No suitable journal found"

    # ---------------- TOP-3 JOURNAL RECOMMENDATIONS ----------------
    top3_recommendations = (
        sorted(
            journal_predictions,
            key=lambda x: (
                round(x.get("confidence", 0.0), 4),
                x.get("journal_name", ""),
            ),
            reverse=True,
        )[:3]
        if journal_predictions
        else []
    )

    # ---------------- EXPLANATION (WHY) ----------------
    rag_explanations = {}

    if False:  # disable RAG for now (too slow and unstable)
        try:
            # Global RAG explanation (already returns parsed JSON)
            parsed_output = _rag_engine.generate(user_abstract, top3_recommendations)

            if not parsed_output:
                parsed_output = {
                    "best_journal": best_journal,
                    "reason": "Selected based on highest similarity and reranker confidence.",
                }

            rag_explanations = {
                "global_explanation": parsed_output,
                "retrieved_papers": None,
            }

            # -------- PER JOURNAL EXPLANATION (fast + stable) --------
            for journal in top3_recommendations:
                journal_name = journal.get("journal_name", "Unknown")
                sim = round(journal.get("similarity", 0.0), 3)

                margin_signal = (
                    raw_scores[0] - raw_scores[1]
                    if len(raw_scores) > 1
                    else raw_scores[0]
                )

                if sim >= 0.75 and margin_signal > 0.2:
                    reason_text = f"{journal_name} is an excellent match with very high semantic similarity ({sim}) and strong ranking confidence, indicating close alignment in research domain and methodology."
                elif sim >= 0.5:
                    reason_text = f"{journal_name} is a reasonable match with moderate similarity ({sim}), suggesting overlap in research topics and application areas."
                else:
                    reason_text = f"{journal_name} has weaker alignment (similarity: {sim}), indicating only partial thematic relevance."

                journal["explanation"] = {
                    "reason": reason_text,
                    "similarity": sim,
                }

        except Exception as e:
            rag_explanations = {"error": str(e)}

    # ---------------- LOGGING FOR FUTURE TRAINING ----------------
    log_entry = {
        "input": user_abstract,
        "predicted_top1": best_journal,
        "confidence": confidence,
        "top3": [j.get("journal_name") for j in top3_recommendations],
    }

    LOG_PATH = os.path.join(BASE_DIR, "data", "prediction_logs.jsonl")

    try:
        with open(LOG_PATH, "a") as f:
            f.write(str(log_entry) + "\n")
    except Exception:
        pass

    return {
        "spread": round(spread, 3),
        "journal_predictions": journal_predictions,
        "top3_recommendations": top3_recommendations,
        "rag_explanations": rag_explanations,
        "final_decision": final_decision,
        "recommended_headings": recommended_headings,
        "submission_recommendation": submission,
        "final_recommendation": final_recommendation,
        "best_journal": best_journal,
        "best_confidence": confidence,
    }
