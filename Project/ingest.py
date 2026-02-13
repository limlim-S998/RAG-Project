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
    """Create stable IDs so repeated ingest updates existing chunks instead of duplicating."""
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
    separators = [
        "\n\n",  # 1st choice: paragraph breaks
        "\n",  # 2nd: line breaks
        ". ",  # 3rd: sentence endings
        "? ",
        "! ",
        "; ",  # 4th: clause boundaries
        ", ",
        " ",  # 5th: word boundaries
        "",  # last resort
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
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    chunk_ids = _build_chunk_ids(chunks)
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest_pdfs()
