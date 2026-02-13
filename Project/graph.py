"""
LangGraph state graph that powers the question-answering pipeline.

The graph is a linear chain of four nodes:

    START --> route --> rewrite --> retrieve --> generate --> END

Each node receives the shared GraphState, updates the fields it owns, and
passes control to the next node.  The separation into distinct nodes keeps
each step focused and makes it straightforward to add branching, retries,
or new nodes later (e.g. a grading step between retrieve and generate).

State fields:
    question         -- the user's question (may be rewritten mid-pipeline)
    documents        -- retrieved context chunks
    answer           -- the final generated answer
    metadata_filter  -- optional Chroma 'where' filter set by the router
"""

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
    """Typed dictionary that every node reads from and writes to."""

    question: str
    documents: list[Document]
    answer: str
    metadata_filter: dict | None


# ── Nodes ─────────────────────────────────────────────


def route(state: GraphState) -> dict:
    """
    Check whether the question mentions a known document title.

    If a title is found, set a metadata filter so that downstream retrieval
    only searches within that document.  Otherwise leave the filter empty
    and let retrieval search across everything.
    """
    titles = get_available_titles()
    question_lower = state["question"].lower()
    for title in titles:
        if title.lower() in question_lower:
            return {"metadata_filter": {"title": title}}
    return {"metadata_filter": None}


def rewrite(state: GraphState) -> dict:
    """
    Ask the LLM to rephrase the question for better retrieval.

    Vague questions like "tell me about that character" get turned into
    something more specific while preserving the original intent.  The
    metadata filter from the routing step is forwarded unchanged.
    """
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
    """
    Run a similarity search against ChromaDB using the (possibly rewritten)
    question.  If a metadata filter was set by the router, retrieval is
    scoped to that specific document.
    """
    retriever = get_retriever(metadata_filter=state.get("metadata_filter"))
    documents = retriever.invoke(state["question"])
    return {"documents": documents}


def generate(state: GraphState) -> dict:
    """
    Feed the retrieved chunks and the question to the LLM and produce a
    final answer.  The system prompt instructs the model to only use the
    provided context and to be upfront when the context doesn't contain
    enough information to answer.
    """
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
    """
    Assemble and compile the LangGraph state graph.

    Returns a compiled graph that can be invoked with:
        result = build_graph().invoke({"question": "..."})
    """
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
