from typing import List, Dict

PRIMARY_THRESHOLD = 0.75
DELTA_THRESHOLD = 0.07   # dominance gap


def recommend_submission_journals(journal_predictions, top_k=3):
    if not journal_predictions:
        return None

    sorted_journals = sorted(
        journal_predictions,
        key=lambda x: x["avg_top_similarity"],
        reverse=True
    )

    best = sorted_journals[0]

    confidence = round(
        0.7 * best["avg_top_similarity"] +
        0.3 * best["max_similarity"],
        3
    )
    confidence = max(0.0, min(confidence, 1.0))

    return {
        "recommended_journal": best["journal_name"],
        "confidence": confidence,
        "reason": f"High similarity across {best['article_matches']} articles",
        "alternatives": [
            j["journal_name"] for j in sorted_journals[1:top_k]
        ]
    }