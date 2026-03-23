from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from src.rag.rag_engine import RAGEngine
import asyncio

# Phase 0
# ---------------- PHASE0 FALLBACK (COMPATIBILITY FIX) ----------------
try:
    from src.phase0.abstract_duplicate_checker import preload_dataset_embeddings
except ImportError:

    def preload_dataset_embeddings(*args, **kwargs):
        print("[Phase0] preload_dataset_embeddings not found, skipping...")


from src.phase0.abstract_duplicate_checker import check_against_dataset

# Phase 1
from src.phase1.title_duplicate_checker import (
    check_title_against_dataset,
    preload_title_embeddings,
)

# Phase 2
from src.phase2.optimized_phase2 import (
    preload_phase2,
    run_phase2_fast,
    make_final_decision,
)

app = FastAPI()

# ----------------------------
# LOAD DATASET ONCE
# ----------------------------

df = pd.read_csv("data/master_journals_expanded.csv")
USE_RAG = False  # Set to True to enable RAG analysis (can be slow on CPU)

# ----------------------------
# INITIALIZE RAG ENGINE (LAZY LOAD)
# ----------------------------
rag_engine = None

if USE_RAG:
    print("Initializing RAG engine...")
    rag_engine = RAGEngine(df)

ALL_JOURNALS = df["journal_name"].dropna().unique().tolist()

DATASET_ABSTRACTS = df["abstract"].dropna().tolist()
DATASET_TITLES = df["article_title"].dropna().tolist()


# ----------------------------
# PRELOAD MODELS ON STARTUP
# ----------------------------


@app.on_event("startup")
def preload_models():
    print("Preloading abstract embeddings...")
    preload_dataset_embeddings(DATASET_ABSTRACTS)

    print("Preloading title embeddings...")
    preload_title_embeddings(DATASET_TITLES)

    print("Preloading Phase 2 FAISS...")
    preload_phase2(df)

    print("Startup complete.")


# ----------------------------
# REQUEST MODELS
# ----------------------------


class TitleCheckRequest(BaseModel):
    title: str


class AnalyzeRequest(BaseModel):
    title: str
    abstract: str
    candidate_journals: Optional[List[str]] = []


# ----------------------------
# PHASE 1 — TITLE VALIDATION
# ----------------------------


@app.post("/check-title")
def check_title(req: TitleCheckRequest):

    title = req.title.strip()

    if not title:
        return {"status": "ERROR", "message": "Title cannot be empty"}

    try:
        result = check_title_against_dataset(title, DATASET_TITLES)
        return result
    except Exception as e:
        return {"status": "ERROR", "message": f"Title check failed: {str(e)}"}


# ----------------------------
# PHASE 0 + PHASE 2 — ABSTRACT ANALYSIS
# ----------------------------


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):

    title = req.title.strip()
    abstract = req.abstract.strip()

    if not title or not abstract:
        return {"status": "ERROR", "message": "Title and abstract are required"}

    try:
        # -------- PHASE 0 --------
        duplication = check_against_dataset(abstract, DATASET_ABSTRACTS)

        if duplication["verdict"] == "DUPLICATE":
            return {
                "status": "EXACT_MATCH",
                "duplication_confidence": duplication["confidence"],
                "message": "Abstract already exists in dataset",
            }

        # -------- PHASE 2 (FAST) --------
        result = run_phase2_fast(user_abstract=abstract, top_k=30)

        # FALLBACK: if aggregation collapses
        if not result.get("journal_predictions") or all(
            j.get("confidence", 0) == 0 for j in result.get("journal_predictions", [])
        ):
            article_results = result.get("article_results", [])
            if article_results:
                top_article = article_results[0]
                result["journal_predictions"] = [
                    {
                        "journal": top_article["journal_name"],
                        "confidence": round(top_article["similarity"], 3),
                    }
                ]

        journal_predictions = result.get("journal_predictions", [])

        submission = result.get(
            "submission_recommendation",
            {"journal": "No suitable journal", "confidence": 0.0},
        )

        confidence = submission.get("confidence", 0.0)
        if confidence == 0.0 and journal_predictions:
            confidence = journal_predictions[0].get("confidence", 0.05)

        best_journal = submission.get("journal") or (
            journal_predictions[0].get("journal", "Unknown")
            if journal_predictions
            else "No suitable journal"
        )

        result["final_recommendation"] = f"{best_journal} (confidence: {confidence})"

        # -------- ADD DIAGNOSTICS (entropy + margin + similarity) --------
        try:
            preds = result.get("journal_predictions", [])

            if preds:
                scores = [j.get("confidence", 0.0) for j in preds]
                sims = [j.get("similarity", 0.0) for j in preds]

                # margin
                sorted_scores = sorted(scores, reverse=True)
                top1 = sorted_scores[0]
                top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
                margin = round(top1 - top2, 4)

                # entropy (robust + NaN safe)
                probs = np.array(scores, dtype=np.float64)

                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    probs = np.ones_like(probs) / max(len(probs), 1)
                else:
                    probs = probs / total

                # remove any NaNs or infs
                probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

                # ensure strictly positive for log
                probs = np.clip(probs, 1e-8, 1.0)

                entropy = -np.sum(probs * np.log(probs))

                if not np.isfinite(entropy):
                    entropy = 0.0

                entropy = round(float(entropy), 4)

                # best similarity
                best_sim = max(sims) if sims else 0.0

                result["diagnostics"] = {
                    "margin": margin,
                    "entropy": entropy,
                    "best_similarity": round(best_sim, 4),
                }
        except Exception as e:
            result["diagnostics"] = {"error": str(e)}

        print(f"[DEBUG] Phase2 best_journal={best_journal} confidence={confidence}")

        # -------- RAG (EXPLANATION) --------
        if USE_RAG and rag_engine is not None:
            try:
                # use top3 journals directly
                top3 = result.get("top3_recommendations", [])[:3]

                parsed_output = await asyncio.to_thread(
                    rag_engine.generate, user_abstract=abstract, top_journals=top3
                )

                result["rag_analysis"] = {"global": parsed_output}

                # attach explanation to each journal
                for journal in top3:
                    journal["explanation"] = {
                        "reason": parsed_output.get("reason", "No explanation"),
                        "similarity": round(journal.get("similarity", 0.0), 3),
                    }

            except Exception as e:
                result["rag_analysis"] = {"error": str(e)}
        else:
            result["rag_analysis"] = None

        # Attach near duplicate warning (non-blocking)
        if duplication["verdict"] == "NEAR_DUPLICATE":
            result["status"] = "NEAR_DUPLICATE"
            result["duplication_confidence"] = duplication["confidence"]

        if "final_recommendation" not in result:
            result["final_recommendation"] = "No suitable journal found"
        return result

    except Exception as e:
        return {"status": "ERROR", "message": f"Analysis failed: {str(e)}"}


# ----------------------------
# HEALTH CHECK
# ----------------------------


@app.get("/health")
def health():
    return {"status": "ok"}
