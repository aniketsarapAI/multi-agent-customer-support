# Self-RAG MCP — Multi-Agent Customer Support

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://multiagent-ai.localnode.app)

A multi-agent AI customer support system combining Self-RAG (document Q&A), SQL querying (TiDB Cloud), hybrid decomposition, and human escalation — all orchestrated by a unified LangGraph. Deployed on AWS EC2 with Docker Compose, Nginx, and HTTPS.

---

## Live Demo

**https://multiagent-ai.localnode.app**

Try it:

| Query | What it does |
|---|---|
| `How many customers do we have?` | SQL → 99,441 |
| `What does Bloomly do?` | RAG → document search |
| `Thanks!` | Conversation → direct reply |
| `Talk to a human` | Escalation → email sent |

---

## Architecture

```
Browser
   │
   ▼
Nginx (HTTPS → HTTP)
   │
   ├── /chat → FastAPI :8000
   │              │
   │              ▼
   │         LangGraph
   │          ├── RAG (FAISS + sentence-transformers)
   │          ├── SQL (TiDB Cloud + PyMySQL)
   │          ├── Hybrid (parallel subgraphs)
   │          └── Escalation (Gmail SMTP)
   │
   └── /*    → Streamlit UI :8501
```

Two Docker containers on a single AWS EC2 instance, connected via Docker networking.

---

## Features

- **Document Q&A** — Self-RAG over company PDFs with retrieval, relevance filtering, support verification, and usefulness checks
- **Database Q&A** — Natural language to SQL against `ecommerce_v2` (11 tables, 1.5M rows) with error recovery
- **Hybrid Questions** — Decompose compound questions spanning docs + database, execute subgraphs in parallel, merge results
- **SQL Visualization** — Auto-detects aggregatable columns and renders Vega-Lite bar charts
- **Query Rewriting** — Vague SQL questions are refined before generation
- **Human Escalation** — Detects unresolved issues via `issup`/`isuse` signals; sends structured HTML email handoff
- **Agent Trace** — Every graph step is logged; the UI renders a formatted execution timeline
- **Rate Limited** — 20 requests/minute/IP via Nginx

---

## Stack

| Layer | Technology |
|---|---|
| LLM | `openai/gpt-4o-mini` via [OpenRouter](https://openrouter.ai) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local CPU) |
| Vector Store | In-memory FAISS |
| Database | TiDB Cloud (`ecommerce_v2`, 11 tables) |
| Graph Framework | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) |
| UI | [Streamlit](https://streamlit.io) |
| Deployment | AWS EC2 (t3.small), Docker Compose, Nginx, Let's Encrypt |

---

## Project Structure

```
selfragmcp/
├── app/
│   ├── api.py                # FastAPI endpoints (/health, /chat)
│   ├── config.py             # Constants and paths
│   ├── state.py              # LangGraph State definition
│   ├── models.py             # Pydantic models for structured output
│   ├── prompts.py            # All LLM prompts
│   ├── llm.py                # OpenRouter LLM client
│   ├── vector_store.py       # FAISS + embedding pipeline
│   ├── db_agent.py           # TiDB Cloud connection
│   ├── email_alert.py        # Gmail SMTP escalation
│   ├── chat_history.py       # History formatting
│   ├── stream_events.py      # Token streaming callback
│   └── graph/
│       ├── builder.py        # LangGraph assembly
│       ├── nodes.py          # RAG nodes (retrieve, generate, verify)
│       ├── rag_subgraph.py   # Compiled RAG subgraph
│       ├── sql_nodes.py      # SQL agent + hybrid decomposition
│       └── escalation.py     # Escalation check + handoff
├── ui/
│   └── streamlit_app.py      # Streamlit frontend (HTTP client)
├── documents/                # Company policy PDFs
├── Dockerfile                # API container
├── Dockerfile.ui             # UI container
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # API dependencies
├── requirements-ui.txt       # UI dependencies
├── .env.example              # Environment template
├── BUILD_LOG.md              # Build journey (1090 lines)
├── DEPLOYMENT.md             # AWS deployment journey (1148 lines)
└── SYSTEM_DESIGN.md          # Full graph architecture
```

---

## Quick Start

### Clone and Configure

```bash
git clone https://github.com/aniketsarapAI/multi-agent-customer-support
cd selfragmcp
cp .env.example .env
# Fill in .env with your API keys (OpenRouter, TiDB, Gmail)
```

### Option A: Docker Compose (recommended)

```bash
docker compose up --build
```

- API: http://localhost:8000 (docs at /docs)
- UI: http://localhost:8501

### Option B: Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

Requires FastAPI running separately:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

---

## API

| Method | Path | Description | Rate Limit |
|---|---|---|---|
| `GET` | `/health` | Health check (ALB ready) | Unlimited |
| `POST` | `/chat` | Ask a question | 20/min |

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many customers?", "chat_history": []}'
```

**Response:**

```json
{
  "answer": "The total count of customers is 99,441.",
  "query_type": "database",
  "logs": ["✅ classify_question: database", "✅ execute_sql_node: 1 row(s)", ...],
  "visualization_spec": null,
  "escalated": false,
  "handoff_summary": "",
  "sql_result": "[{\"count(*)\": 99441}]"
}
```

Full API docs at http://localhost:8000/docs (Swagger UI).

---

## Deployment

Production: **https://multiagent-ai.localnode.app** (AWS EC2, Docker Compose, Nginx, HTTPS)

The full 25-phase deployment journey is documented in **[DEPLOYMENT.md](DEPLOYMENT.md)**, covering:

1. Architecture refactor (Streamlit → HTTP client)
2. API contract expansion
3. Containerization (Dockerfile, Dockerfile.ui)
4. EC2 provisioning (t3.small, security groups)
5. Docker Buildx/BuildKit troubleshooting
6. Elastic IP + custom domain
7. Nginx reverse proxy + Let's Encrypt SSL

---

## Project Journey

This project has two parallel narrative documents tracing the complete arc from prototype to production:

- **[BUILD_LOG.md](BUILD_LOG.md)** (1090 lines) — First-person account of building the LangGraph application: prompt engineering, graph routing, SQL agent design, and the iteration cycle.
- **[DEPLOYMENT.md](DEPLOYMENT.md)** (1148 lines) — Full AWS deployment journey: architecture refactor, API contract expansion, containerization, EC2 setup, Buildx troubleshooting, domain configuration, Nginx, and HTTPS.

For the system architecture with mermaid diagrams, see **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)**.

---

## Key Lessons

1. **Separate UI from business logic.** Streamlit initially imported LangGraph directly. Refactoring to HTTP-only communication enabled clean deployment.
2. **Frontends should never depend on backend internals.** Fields like `rag_docs_used`, `sql_result`, and `handoff_summary` had to be added to the API contract because the UI was reading them from graph state.
3. **Docker builds are environment-dependent.** A working local build does not guarantee AWS success — Buildx and BuildKit compatibility matters.
4. **Start simple.** EC2 + Docker Compose was the right first deployment. Premature migration to ECS Fargate would have added complexity without proportional benefit.
5. **Always create billing safeguards.** A $5 monthly budget alert prevents surprise charges.
