STRONG_MATCH_THRESHOLD = 0.75
WEAK_MATCH_THRESHOLD = 0.45


def clamp(score: float) -> float:
    return max(0.0, min(1.0, round(float(score), 3)))


def make_final_decision(journal_predictions, semantic_validation=None):
    """
    Always returns structured dict:
    {
        "journal": str,
        "confidence": float
    }
    """

    if not journal_predictions:
        return {
            "journal": "No suitable journal",
            "confidence": 0.0,
        }

    top = journal_predictions[0]
    journal_name = top.get("journal_name", "Unknown")

    # 🔴 YOUR BUG WAS HERE
    # You were using max_similarity (does not exist)
    # DO NOT reprocess confidence — trust normalized values
    top_confidence = float(top.get("confidence", 0.0))
    # ---------- SEMANTIC OVERRIDE ----------
    if semantic_validation:
        topic_alignment = clamp(semantic_validation.get("topic_alignment", 0.0))
        embedding_similarity = clamp(
            semantic_validation.get("embedding_similarity", 0.0)
        )
        techniques = semantic_validation.get("techniques", [])

        strong_technique = any(t.get("confidence", 0) >= 0.85 for t in techniques)

        if topic_alignment >= 0.8 and embedding_similarity < 0.6 and strong_technique:
            return {
                "journal": journal_name,
                "confidence": max(top_confidence, 0.65),
            }

    # ---------- NORMAL DECISION ----------
    return {
        "journal": journal_name,
        "confidence": top_confidence,
    }
