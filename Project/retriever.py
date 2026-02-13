try:
    from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K
except ImportError:
    from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

from langchain_chroma import Chroma  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def get_retriever(metadata_filter: dict | None = None):
    vectorstore = get_vectorstore()
    search_kwargs = {"k": TOP_K}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def get_available_titles() -> list[str]:
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    all_meta = collection.get(include=["metadatas"])["metadatas"] or []
    return sorted({str(m["title"]) for m in all_meta if "title" in m})
