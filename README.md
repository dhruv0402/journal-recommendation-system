# Journal Recommendation System

An AI-powered system that recommends academic journals based on research abstracts using semantic search and RAG.

## Features
- FAISS-based semantic search
- SentenceTransformer embeddings
- Multi-stage ranking pipeline
- RAG-based explanation using Groq LLM
- FastAPI backend + Streamlit frontend
- Sub-300ms inference

## Tech Stack
- Python
- FAISS
- SentenceTransformers
- FastAPI
- Streamlit
- Groq API

## Run Locally

### Backend
uvicorn main:app --reload

### Frontend
streamlit run app.py