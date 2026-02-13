from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / "data" / "pdfs"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# ── Embedding model (HuggingFace, runs locally) ───────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── LLM (Ollama, runs locally) ────────────────────────
OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Retrieval ──────────────────────────────────────────
TOP_K = 5

# ── ChromaDB ───────────────────────────────────────────
COLLECTION_NAME = "pdf_documents"
