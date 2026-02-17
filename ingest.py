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

import hashlib
from collections import defaultdict
from pathlib import Path

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_DIR,
)

from langchain_chroma import Chroma  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore


def _build_chunk_ids(chunks: list[Document]) -> list[str]:
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


def _discover_pdfs() -> list[Path]:
    """Find all PDF files in the configured data directory."""
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}")
    return pdf_paths


def _load_and_split(pdf_paths: list[Path]) -> list[Document]:
    """
    Load every page from the given PDFs and split them into overlapping
    text chunks.

    The separator list is ordered from most to least preferred.  The splitter
    tries the first separator, and only falls back to the next one when the
    resulting chunk would exceed CHUNK_SIZE.
    """

    documents: list[Document] = []

    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDF(s)")

    separators = [
        "\n\n",  # paragraph breaks
        "\n",  # line breaks
        ". ",  # sentence endings
        "? ",
        "! ",
        "; ",  # clause boundaries
        ", ",
        " ",  # word boundaries
        "",  # character-level (last resort)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def _store_chunks(chunks: list[Document], reset_collection: bool) -> Chroma:
    """Embed chunks with HuggingFace and store them in ChromaDB."""
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
        except ValueError:
            pass
        # Re-create the handle after deleting, since the old one points
        # to a now-deleted collection.
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    chunk_ids = _build_chunk_ids(chunks)
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")
    return vectorstore


def ingest_pdfs(reset_collection: bool = True) -> Chroma | None:
    """
    Main ingestion entrypoint.

    Discovers PDFs, loads and splits them into chunks, then embeds and
    stores them in ChromaDB.

    Args:
        reset_collection: If True (default), the existing Chroma collection
            is deleted before inserting new chunks.  Set to False if you want
            to append new documents without removing old ones.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = _discover_pdfs()
    if not pdf_paths:
        return None

    chunks = _load_and_split(pdf_paths)
    return _store_chunks(chunks, reset_collection)


if __name__ == "__main__":
    ingest_pdfs()
