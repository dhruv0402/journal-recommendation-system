import pandas as pd
from src.phase2.semantic_similarity import compute_structured_similarity
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
df = pd.read_csv("data/master_journals_enriched.csv")

def get_semantics(row):
    return {
        "domain": row["domain"],
        "techniques": str(row["techniques"]).split(","),
        "keywords": str(row["keywords"]).split(","),
    }

scores = []

journals = df["journal_name"].unique()

for j in journals[:10]:  # limit for speed
    subset = df[df["journal_name"] == j].head(5)

    for i in range(len(subset)):
        for k in range(i + 1, len(subset)):
            s1 = get_semantics(subset.iloc[i])
            s2 = get_semantics(subset.iloc[k])
            score = compute_structured_similarity(s1, s2)
            scores.append(("same", score))

# Cross-journal comparisons
for j1 in journals[:5]:
    for j2 in journals[5:10]:
        s1 = get_semantics(df[df["journal_name"] == j1].iloc[0])
        s2 = get_semantics(df[df["journal_name"] == j2].iloc[0])
        score = compute_structured_similarity(s1, s2)
        scores.append(("cross", score))

result = pd.DataFrame(scores, columns=["type", "score"])
print(result.groupby("type")["score"].describe())