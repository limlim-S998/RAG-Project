# RAG_Lang

A local retrieval-augmented generation (RAG) pipeline for querying PDF documents. Built on LangChain and LangGraph, it runs entirely on your machine — no API keys or cloud services required.

Drop PDFs into a folder, ingest them into a ChromaDB vector store, and ask questions through a multi-step LangGraph pipeline that routes, rewrites, retrieves, and generates answers using Ollama.

## How It Works
The pipeline is structured as a LangGraph state graph with four sequential nodes:

```
Route --> Rewrite --> Retrieve --> Generate
```

1. **Route** — Scans the question for known document titles. If one is found, a metadata filter is set so retrieval is scoped to that specific document.
2. **Rewrite** — Sends the question through the LLM to make it more specific and better suited for semantic search, while preserving the original intent.
3. **Retrieve** — Queries ChromaDB for the top-k most relevant chunks using HuggingFace sentence embeddings, optionally filtered by the metadata from the routing step.
4. **Generate** — Feeds the retrieved context and original question to the LLM, which produces a grounded answer. If the context doesn't contain the answer, the model says so.

## Project Structure

```
RAG_Lang/
├── Project/
│   ├── config.py        # Paths, model names, chunking parameters
│   ├── ingest.py        # PDF loading, chunking, embedding, ChromaDB storage
│   ├── retriever.py     # Vector store access and retrieval logic
│   ├── graph.py         # LangGraph state graph definition
│   └── main.py          # CLI entrypoint
├── data/
│   └── pdfs/            # Place your PDF files here
├── requirements.txt
└── .gitignore
```

## Prerequisites

- **Python 3.13+**
- **Ollama** installed and running locally ([ollama.com](https://ollama.com))
- The `llama3:8b` model pulled in Ollama:
  ```bash
  ollama pull llama3:8b
  ```

## Setup

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/<your-username>/RAG_Lang.git
cd RAG_Lang
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All commands are run through `main.py`.

### 1. Ingest PDFs

Place your PDF files in `data/pdfs/`, then run:

```bash
python Project/main.py ingest
```

This loads every PDF, splits the text into overlapping chunks (1500 chars, 300 overlap), embeds them with `all-MiniLM-L6-v2`, and stores the vectors in a local ChromaDB instance under `data/chroma_db/`.

Re-running ingest clears the existing collection by default, so the store always reflects the current contents of the PDF folder.

### 2. Ask a single question

```bash
python Project/main.py ask "What happens to Lucy in Dracula?"
```

### 3. Interactive chat

```bash
python Project/main.py chat
```

Opens a loop where you can ask multiple questions in sequence. Type `quit` to exit.

## Configuration

All tunable parameters live in [Project/config.py](Project/config.py):

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1500 | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | 300 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer model |
| `OLLAMA_MODEL` | `llama3:8b` | Ollama model used for rewriting and generation |
| `TOP_K` | 5 | Number of chunks retrieved per query |

To use a different Ollama model, change `OLLAMA_MODEL` and make sure you've pulled it:

```bash
ollama pull <model-name>
```

## Key Dependencies

- [LangChain](https://github.com/langchain-ai/langchain) — document loading, text splitting, prompt templates
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful graph orchestration
- [ChromaDB](https://github.com/chroma-core/chroma) — local vector storage and similarity search
- [HuggingFace Sentence Transformers](https://github.com/UKPLab/sentence-transformers) — embedding model
- [Ollama](https://ollama.com) — local LLM inference

## License

This project does not currently include a license. If you intend to distribute or share it, consider adding one.
