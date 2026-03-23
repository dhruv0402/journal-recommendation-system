import pandas as pd
import matplotlib.pyplot as plt

from src.phase2.multi_journal_pipeline import run_phase2_multi_journal

# ---------------- CONFIG ----------------
DATA_PATH = "data/master_journals_enriched.csv"

CANDIDATE_JOURNALS = [
    {"journal_name": "Computer Networks"},
    {"journal_name": "IEEE Transactions on AI"},
    {"journal_name": "Bioinformatics"},
]

USER_ABSTRACT = """
This paper studies graph spanner constructions for optimizing
routing efficiency in large-scale communication networks.
Latency and bandwidth trade-offs are evaluated.
"""

# ---------------- RUN ----------------
df = pd.read_csv(DATA_PATH)

result = run_phase2_multi_journal(
    user_abstract=USER_ABSTRACT,
    candidate_journals=CANDIDATE_JOURNALS,
    df=df
)

print("\nFINAL RESULT:\n")
print(result)

# ---------------- CONFIDENCE BAR CHART ----------------
journals = [j["journal"] for j in result["ranked_journals"]]
confidences = [j["confidence"] for j in result["ranked_journals"]]

plt.figure()
plt.bar(journals, confidences)
plt.ylim(0, 1)

# Threshold lines
plt.axhline(0.65, linestyle="--", label="Strong Threshold (0.65)")
plt.axhline(0.45, linestyle="--", label="Partial Threshold (0.45)")

plt.title("Journal Confidence Scores")
plt.ylabel("Confidence")
plt.xlabel("Journal")
plt.legend()

plt.tight_layout()
plt.savefig("journal_confidence.png")
plt.show()