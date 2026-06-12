# Self-RAG MCP

A modular Python application that combines Self-RAG (Retrieval-Augmented Generation) with direct SQL query execution against a TiDB Cloud e-commerce database, all orchestrated by a unified LangGraph.

## Features

- **Document Q&A** — Self-RAG loop over company policy PDFs with retrieval, relevance filtering, support verification, and usefulness checks
- **Database Q&A** — Natural language to SQL using GPT-4o-mini against `ecommerce_v2` (11 tables, 1.5M rows), with error recovery
- **Hybrid Questions** — Decompose compound questions spanning docs + database, execute subgraphs in parallel via `Send`, and merge results
- **Query Rewriting** — Vague SQL questions are refined before SQL generation
- **SQL Visualization** — Auto-detects numeric + categorical columns and renders Vega-Lite bar charts
- **Thinking Logs** — Every node emits progressive log entries; the UI shows the last 12 logs in real time
- **Token Streaming** — LLM tokens are streamed via LangChain callbacks; CLI and Streamlit display tokens progressively

## Stack

| Layer | Technology |
|---|---|
| LLM | `openai/gpt-4o-mini` via [OpenRouter](https://openrouter.ai) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | In-memory FAISS |
| Database | TiDB Cloud (`ecommerce_v2`) |
| Graph Framework | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| UI | [Streamlit](https://streamlit.io) |
| SQL Client | PyMySQL with SSL |

## Setup

```bash
git clone <repo>
cd selfragmcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY + DB credentials
```

## Usage

### CLI
```bash
python main.py "What is the refund policy?"
python main.py -i   # interactive mode
```

### Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## Architecture

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for the full graph architecture with mermaid diagram.
