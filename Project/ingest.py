"""
PDF ingestion pipeline.

Reads every PDF from the configured data directory, splits each document
into overlapping text chunks, generates embeddings via a local HuggingFace
model, and stores everything in a ChromaDB collection.

The pipeline is idempotent: each chunk gets a deterministic ID derived from
its source file, page number, and position on that page.  Re-running ingest
with reset_collection=True (the default) wipes the collection first so the
store always mirrors the current contents of the PDF folder.

Typical usage:
    python Project/main.py ingest
    # or directly:
    python Project/ingest.py
"""

from collections import defaultdict
import hashlib

try:
    from .config import (
        CHROMA_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        PDF_DIR,
    )
except ImportError:
    from config import (
        CHROMA_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        PDF_DIR,
    )

from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore


def _build_chunk_ids(chunks) -> list[str]:
    """
    Create deterministic IDs for each chunk so that repeated ingestion
    upserts (updates-or-inserts) instead of creating duplicates.

    The ID is a SHA-1 hash of "source_path|page_number|ordinal", where
    ordinal is the chunk's index within that specific page.  This means
    the same PDF content always maps to the same ID, regardless of when
    or how many times you run ingest.
    """
    per_page_counter: defaultdict[tuple[str, str], int] = defaultdict(int)
    ids: list[str] = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown_source"))
        page = str(chunk.metadata.get("page", "unknown_page"))
        key = (source, page)
        ordinal = per_page_counter[key]
        per_page_counter[key] += 1

        raw_id = f"{source}|{page}|{ordinal}"
        ids.append(hashlib.sha1(raw_id.encode("utf-8")).hexdigest())

    return ids


def ingest_pdfs(reset_collection: bool = True):
    """
    Main ingestion entrypoint.

    Steps:
        1. Discover all .pdf files in PDF_DIR.
        2. Load every page as a LangChain Document (via PyPDFLoader).
        3. Split pages into smaller, overlapping text chunks using a
           RecursiveCharacterTextSplitter with a hierarchy of separators
           (paragraphs > lines > sentences > words) so splits happen at
           natural boundaries whenever possible.
        4. Embed chunks with HuggingFace and store them in ChromaDB.

    Args:
        reset_collection: If True (default), the existing Chroma collection
            is deleted before inserting new chunks.  Set to False if you want
            to append new documents without removing old ones.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Find all PDFs
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}")
        return

    # 2. Load every page from every PDF into Documents
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDF(s)")

    # 3. Split into chunks
    # The separator list is ordered from most to least preferred. The splitter
    # tries the first separator, and only falls back to the next one when the
    # resulting chunk would exceed CHUNK_SIZE.
    separators = [
        "\n\n",  # paragraph breaks
        "\n",    # line breaks
        ". ",    # sentence endings
        "? ",
        "! ",
        "; ",    # clause boundaries
        ", ",
        " ",     # word boundaries
        "",      # character-level (last resort)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 4. Embed and store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    if reset_collection:
        try:
            vectorstore.delete_collection()
            print(f"Cleared existing collection '{COLLECTION_NAME}' before ingest")
        except Exception:
            pass
        # Re-create the vectorstore handle after deleting the collection,
        # since the old handle points to a now-deleted collection.
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    chunk_ids = _build_chunk_ids(chunks)
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    # Older versions of langchain-chroma expose a .persist() method for
    # flushing to disk; newer versions persist automatically.
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest_pdfs()
