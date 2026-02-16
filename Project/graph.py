"""
LangGraph state graph that powers the question-answering pipeline.

The graph is a chain of five nodes with a conditional retry loop:

    START --> route --> rewrite --> retrieve --> generate --> grade --+--> END
                                     ^                              |
                                     |                              |
                                     +--- (fail & retries < 3) ----+

Each node receives the shared GraphState, updates the fields it owns, and
passes control to the next node.

State fields:
    question          -- the user's question (may be rewritten mid-pipeline)
    original_question -- preserved copy of the original question for re-rewrites
    documents         -- retrieved context chunks
    answer            -- the final generated answer
    metadata_filter   -- optional Chroma 'where' filter set by the router
    retries           -- number of retry attempts so far
    _grade_passed     -- whether the last grading check passed
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

MAX_RETRIES = 3

# ── State flowing through the graph ───────────────────


class GraphState(TypedDict):
    """Typed dictionary that every node reads from and writes to."""

    question: str
    original_question: str
    documents: list[Document]
    answer: str
    metadata_filter: dict | None
    retries: int
    _grade_passed: bool


# ── Nodes ─────────────────────────────────────────────


def route(state: GraphState) -> dict:
    """
    Check whether the question mentions a known document title.

    Also initialises retry counter and preserves the original question
    so that retry rewrites don't compound drift.
    """
    titles = get_available_titles()
    question_lower = state["question"].lower()
    metadata_filter = None
    for title in titles:
        if title.lower() in question_lower:
            metadata_filter = {"title": title}
            break
    return {
        "metadata_filter": metadata_filter,
        "original_question": state["question"],
        "retries": 0,
    }


def rewrite(state: GraphState) -> dict:
    """
    Ask the LLM to rephrase the question for better retrieval.

    On retries we rewrite from the *original* question so the LLM
    doesn't compound drift from rewriting a rewrite of a rewrite.
    """
    source_question = state.get("original_question", state["question"])

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
    rewritten = chain.invoke({"question": source_question})
    return {"question": rewritten.content}


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
    final answer.
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


def grade(state: GraphState) -> dict:
    """
    Ask the LLM to judge whether the generated answer actually addresses
    the question using the retrieved context.

    Stores the verdict and increments the retry counter so the conditional
    edge can decide whether to loop or finish.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing the quality of an answer to a question.\n"
                "You will be given the question, the retrieved context, and the generated answer.\n\n"
                "Judge whether the answer:\n"
                "  1. Actually addresses the question asked\n"
                "  2. Is supported by the retrieved context (not hallucinated)\n"
                "  3. Is a substantive response (not just 'I don't know' when context exists)\n\n"
                "Respond with ONLY 'yes' if the answer is acceptable, or 'no' if it should be retried.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Context:\n{context}\n\n"
                "Answer: {answer}\n\n"
                "Is this answer acceptable? (yes/no):",
            ),
        ]
    )

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    chain = prompt | llm
    verdict = chain.invoke(
        {
            "question": state["question"],
            "context": context,
            "answer": state["answer"],
        }
    )

    passed = "yes" in verdict.content.strip().lower()
    retries = state.get("retries", 0) + 1

    print(
        f"  [grade] verdict={'pass' if passed else 'fail'}, attempt {retries}/{MAX_RETRIES}"
    )

    return {"retries": retries, "_grade_passed": passed}


# ── Conditional routing ───────────────────────────────


def decide_after_grade(state: GraphState) -> str:
    """
    Conditional edge after the grade node.

    Routes to END if the answer passed grading or retries are exhausted.
    Routes back to rewrite to try again otherwise.
    """
    if state["_grade_passed"]:
        return "accept"
    if state["retries"] >= MAX_RETRIES:
        print(f"  [grade] max retries ({MAX_RETRIES}) reached, returning best answer")
        return "accept"
    return "retry"


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
    graph.add_node("grade", grade)

    graph.add_edge(START, "route")
    graph.add_edge("route", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")

    # Conditional edge: grade decides whether to accept or retry
    graph.add_conditional_edges(
        "grade",
        decide_after_grade,
        {
            "accept": END,
            "retry": "rewrite",  # loop back for a fresh rewrite + retrieve + generate
        },
    )

    return graph.compile()
