"""
Retriever interface for the vector store.

This module sits between the ChromaDB storage layer and the rest of the
pipeline.  It exposes three things:

    - get_vectorstore()      -- raw Chroma handle (for when you need direct access)
    - get_retriever()        -- a LangChain Retriever ready for use in chains/graphs,
                                with optional metadata filtering
    - get_available_titles() -- a list of distinct document titles currently in the
                                collection, used by the routing node to decide whether
                                a query targets a specific document
"""

try:
    from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K
except ImportError:
    from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

from langchain_chroma import Chroma  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

# Cache the embeddings instance so the model is only loaded once per process
# rather than on every call to get_vectorstore().
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    """Return a Chroma vectorstore handle backed by the on-disk collection."""
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def get_retriever(metadata_filter: dict | None = None):
    """
    Build a LangChain retriever that returns the top-k most similar chunks.

    Args:
        metadata_filter: Optional Chroma 'where' filter, e.g. {"title": "Dracula"}.
            When provided, only chunks whose metadata matches the filter are
            considered during similarity search.  This is how the graph's
            routing node narrows results to a single document.
    """
    vectorstore = get_vectorstore()
    search_kwargs = {"k": TOP_K}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def get_available_titles() -> list[str]:
    """
    Query the Chroma collection for every distinct 'title' value stored in
    chunk metadata.  The routing node uses this list to check whether the
    user's question mentions a known document title.
    """
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    all_meta = collection.get(include=["metadatas"])["metadatas"] or []
    return sorted({str(m["title"]) for m in all_meta if "title" in m})
