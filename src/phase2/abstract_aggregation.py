from collections import defaultdict


def aggregate_abstract_results(article_results, top_k_per_journal=5):
    """
    Top-K weighted aggregation (strong matches dominate)
    """

    journal_scores = defaultdict(list)

    for r in article_results:
        journal_scores[r["journal_name"]].append(r["similarity"])

    aggregated = []

    for journal, scores in journal_scores.items():
        scores = sorted(scores, reverse=True)

        # Take top-k strongest signals
        top_scores = scores[:top_k_per_journal]

        # Weighted scoring (rank-based decay)
        weighted_score = sum(s * (1 / (i + 1)) for i, s in enumerate(top_scores))

        # Calibrated normalization (prevents score saturation)
        weight_norm = sum(1 / (i + 1) for i in range(len(top_scores)))

        # Scale down magnitude BEFORE normalization to avoid dominance
        scaled_score = weighted_score / (1 + 0.3 * (len(top_scores) - 1))
        normalized_score = scaled_score / weight_norm if weight_norm > 0 else 0

        # Soft cap (prevents 1.0 spikes)
        normalized_score = min(0.85, normalized_score)

        aggregated.append(
            {
                "journal_name": journal,
                "confidence": round(float(normalized_score), 3),
                "similarity": round(float(sum(top_scores) / len(top_scores)), 3)
                if top_scores
                else 0.0,
                "top_match": round(float(top_scores[0]), 3) if top_scores else 0.0,
                "matches_used": len(top_scores),
            }
        )

    return sorted(aggregated, key=lambda x: x["confidence"], reverse=True)
