"""
CLI entrypoint for the RAG pipeline.

Provides three commands:
    ingest  -- load PDFs into the vector store (run this first)
    ask     -- send a single question through the pipeline and print the answer
    chat    -- start an interactive loop for asking multiple questions

Examples:
    python Project/main.py ingest
    python Project/main.py ask "Who is Count Dracula?"
    python Project/main.py chat
"""

import argparse

from graph import build_graph
from ingest import ingest_pdfs


def ask(question: str) -> None:
    """Run a single question through the graph and print the result."""
    app = build_graph()
    result = app.invoke({"question": question})
    print(f"\nAnswer:\n{result['answer']}")


def chat() -> None:
    """
    Interactive chat loop.

    The graph is compiled once at the start so model loading only happens on
    the first invocation.  Each question is independent — there is no
    conversation memory between turns.
    """
    app = build_graph()
    print("RAG Chat (type 'quit' to exit)")
    print("-" * 40)
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        result = app.invoke({"question": question})
        print(f"\nAssistant: {result['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Process PDFs into the vector store")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", nargs="+", help="The question to ask")

    subparsers.add_parser("chat", help="Interactive chat loop")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_pdfs()
    elif args.command == "ask":
        ask(" ".join(args.question))
    elif args.command == "chat":
        chat()


if __name__ == "__main__":
    main()
