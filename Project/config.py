"""
Central configuration for the RAG pipeline.

All tunable parameters — paths, model names, chunking behaviour, retrieval
settings — live here so the rest of the codebase can import them without
hard-coding values.  Edit this file to swap models, change chunk sizes, or
point at a different data directory.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / "data" / "pdfs"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

# ── Chunking ───────────────────────────────────────────
# CHUNK_SIZE controls the max character length of each text fragment.
# CHUNK_OVERLAP ensures neighbouring chunks share some context at their
# boundaries, which helps the retriever surface passages that straddle a
# chunk boundary.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# ── Embedding model (HuggingFace, runs locally) ───────
# all-MiniLM-L6-v2 is a lightweight sentence-transformer that runs on CPU.
# Swap this for a larger model if you need higher retrieval accuracy and
# have the hardware for it.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── LLM (Ollama, runs locally) ────────────────────────
# Requires Ollama to be running (https://ollama.com).
# Pull the model first:  ollama pull llama3:8b
OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Retrieval ──────────────────────────────────────────
# Number of document chunks returned per query.
TOP_K = 5

# ── ChromaDB ───────────────────────────────────────────
# Name of the Chroma collection. Changing this effectively creates a
# separate vector store, so make sure to re-ingest if you rename it.
COLLECTION_NAME = "pdf_documents"
