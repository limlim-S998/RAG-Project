import sys

try:
    from .graph import build_graph
    from .ingest import ingest_pdfs
except ImportError:
    from graph import build_graph
    from ingest import ingest_pdfs


def ask(question: str):
    app = build_graph()
    result = app.invoke({"question": question})
    print(f"\nAnswer:\n{result['answer']}")


def chat():
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


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest        — Process PDFs into the vector store")
        print("  python main.py ask <question> — Ask a single question")
        print("  python main.py chat           — Interactive chat loop")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        ingest_pdfs()
    elif command == "ask":
        if len(sys.argv) < 3:
            print('Provide a question: python main.py ask "your question here"')
            sys.exit(1)
        ask(" ".join(sys.argv[2:]))
    elif command == "chat":
        chat()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
