import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq


class RAGEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # Faster embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

        # cache for query embeddings to avoid recomputation
        self.query_cache = {}

        # -------- BUILD DATASET EMBEDDINGS (BATCHED) --------
        abstracts = df["abstract"].fillna("").tolist()

        import os

        cache_path = "data/rag_embeddings.npy"
        index_path = "data/rag_faiss.index"

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path).astype("float32")
        else:
            embeddings = self.model.encode(
                abstracts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.embeddings = np.array(embeddings).astype("float32")
            np.save(cache_path, self.embeddings)

        dim = self.embeddings.shape[1]

        # -------- FAISS IVF INDEX (CACHED ON DISK FOR FAST STARTUP) --------
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            nlist = 100
            quantizer = faiss.IndexFlatIP(dim)

            self.index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )

            if not self.index.is_trained:
                self.index.train(self.embeddings)

            self.index.add(self.embeddings)
            self.index.nprobe = 10

            # save index so next startup is instant
            faiss.write_index(self.index, index_path)

        # -------- GROQ LLM (SAFE INIT) --------
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in your environment variables."
            )

        self.client = Groq(api_key=api_key)

    def retrieve(self, query: str, k: int = 5):

        # ---- QUERY EMBEDDING CACHE ----
        if query in self.query_cache:
            query_vec = self.query_cache[query]
        else:
            query_vec = self.model.encode([query], normalize_embeddings=True)
            query_vec = np.array(query_vec).astype("float32")
            self.query_cache[query] = query_vec

        if not self.index.is_trained:
            self.index.train(self.embeddings)

        distances, indices = self.index.search(query_vec, k)

        papers = []

        for idx in indices[0]:
            if idx < len(self.df):
                row = self.df.iloc[idx]

                papers.append(
                    {"journal": row["journal_name"], "abstract": row["abstract"]}
                )

        return papers

    def generate(self, user_abstract: str, top_journals):
        """
        Generate explanation using top-ranked journals instead of random retrieval.
        """
        if not top_journals:
            return {
                "best_journal": "Unknown",
                "reason": "No journal candidates provided",
            }

        context_lines = []
        for i, j in enumerate(top_journals[:3]):
            journal_name = j.get("journal_name", "Unknown Journal")
            similarity = round(j.get("similarity", 0.0), 3)

            context_lines.append(f"{i + 1}. {journal_name} (similarity: {similarity})")

        context = "\n".join(context_lines)

        prompt = f"""
You are an expert academic journal recommender.

Abstract:
{user_abstract}

Top candidate journals:
{context}

Task:
- Select the BEST journal
- Explain WHY it fits

STRICT RULES:
- Output ONLY valid JSON
- No extra text
- Keep explanation under 3 sentences

JSON format:
{{
  "best_journal": "string",
  "reason": "string"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            output = response.choices[0].message.content.strip()
        except Exception:
            return {
                "best_journal": top_journals[0].get("journal_name", "Unknown"),
                "reason": "LLM failed to generate explanation",
            }

        import json

        try:
            start = output.find("{")
            end = output.rfind("}") + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found")

            parsed = json.loads(output[start:end])

            if "best_journal" not in parsed:
                parsed["best_journal"] = top_journals[0].get("journal_name", "Unknown")

            if "reason" not in parsed:
                parsed["reason"] = "Reason not provided"

            return parsed

        except Exception:
            return {
                "best_journal": top_journals[0].get("journal_name", "Unknown"),
                "reason": output[:200],
            }
