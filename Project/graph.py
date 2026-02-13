from typing import TypedDict

try:
    from .config import OLLAMA_BASE_URL, OLLAMA_MODEL
    from .retriever import get_available_titles, get_retriever
except ImportError:
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL
    from retriever import get_available_titles, get_retriever

from langchain_core.documents import Document  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_ollama import ChatOllama  # type: ignore
from langgraph.graph import END, START, StateGraph  # type: ignore


# ── State flowing through the graph ───────────────────
class GraphState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    metadata_filter: dict | None


# ── Nodes ─────────────────────────────────────────────
def route(state: GraphState) -> dict:
    titles = get_available_titles()
    question_lower = state["question"].lower()
    for title in titles:
        if title.lower() in question_lower:
            return {"metadata_filter": {"title": title}}
    return {"metadata_filter": None}


def rewrite(state: GraphState) -> dict:
    # We want to rewrite the question to be more specific, but we don't want
    # to lose the metadata filter info from the previous node. So we return a
    # dict with both keys.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that rewrites questions to be more specific. "
                "Rewrite the question to be more specific, but keep the meaning the same. "
                "If the question is already specific, just return it as is.",
            ),
            ("human", "Question: {question}\n\nRewrite:"),
        ]
    )
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    chain = prompt | llm
    rewritten = chain.invoke({"question": state["question"]})
    return {"question": rewritten.content, "metadata_filter": state["metadata_filter"]}


def retrieve(state: GraphState) -> dict:
    retriever = get_retriever(metadata_filter=state.get("metadata_filter"))
    documents = retriever.invoke(state["question"])
    return {"documents": documents}


def generate(state: GraphState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions based on the "
                "provided context. Only use the context below to answer. If the "
                "context does not contain the answer, say so.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}


# ── Build the graph ───────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "route")
    graph.add_edge("route", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
