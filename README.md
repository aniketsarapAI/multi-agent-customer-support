# Self-RAG MCP — Multi-Agent Customer Support

AI-powered customer support that knows both your documents *and* your database. This project orchestrates three decision paths—document retrieval, SQL querying, and hybrid decomposition—through a unified LangGraph. Every question is routed intelligently. Every failure gracefully escalates to humans.

<p align="center">
  <img src="assets/hero.svg" alt="Self-RAG MCP System Blueprint" width="100%" />
</p>

## Table of Contents

- [Tech Stack](#-core-tech-stack)
- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Features](#features)
- [Engineering Decisions](#engineering-decisions)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Testing & Validation](#testing--validation)
- [Roadmap](#roadmap)

## 🛠️ Core Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python_3.12-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/FastAPI-109989?style=for-the-badge&logo=FASTAPI&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/TiDB_Cloud-FF6C37?style=for-the-badge&logo=tidb&logoColor=white" alt="TiDB Cloud" />
</p>

<p align="left">
  <img src="https://img.shields.io/badge/FAISS-1C3C3C?style=for-the-badge&logo=vector&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/OpenRouter-0055DA?style=for-the-badge&logo=rapid&logoColor=white" alt="OpenRouter" />
  <img src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS EC2" />
  <img src="https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white" alt="Nginx" />
</p>

## Live Demo

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://multiagent-ai.localnode.app)


**Try these queries:**

| Query | Agent | Result |
|---|---|---|
| `How many customers do we have?` | SQL → TiDB | 99,441 |
| `What does Bloomly do?` | RAG → FAISS | Company description |
| `Thanks!` | Conversation | Direct response |
| `Talk to a human` | Escalation | Gmail handoff |
| `How many customers from the US with orders over $500?` | Hybrid | Document + SQL |

## Project Overview

Self-RAG MCP is a deployment-ready multi-agent customer support system combining:

✅ **Document Q&A** — Self-RAG over company PDFs with retrieval, relevance filtering, and support verification  
✅ **SQL Agent** — Natural language to SQL against TiDB Cloud with automatic error recovery  
✅ **Hybrid Decomposition** — Complex questions split into parallel sub-agents, merged intelligently  
✅ **Human Escalation** — Unresolved issues detected and routed to Gmail with structured context  
✅ **Query Intelligence** — Automatic classification and routing without explicit user direction  
✅ **Structured Logging** — Every graph step traced and rendered in the UI timeline  
✅ **API-First Architecture** — Separate API from UI, cleanly deployable to Docker + AWS  
✅ **Production Deployment** — EC2 + Docker Compose + Nginx + Let's Encrypt HTTPS  

## Features

<p align="center">
  <img src="assets/features.svg" alt="Multi-Agent Capability Mesh" width="100%" />
</p>

### Core Intelligence

**Document Retrieval (RAG)**  
Uses Self-RAG pattern: retrieve → generate → verify → usefulness check. FAISS vector store with `sentence-transformers/all-MiniLM-L6-v2` embeddings. Relevance filtering with configurable thresholds. Loops if usefulness check fails (max 10 retries).

**SQL Agent**  
Converts natural language to SQL against TiDB Cloud (`ecommerce_v2`, 11 tables, 1.5M rows). Automatic error recovery: failed queries are refined using error messages. Detects aggregatable columns and auto-generates Vega-Lite bar charts. Fallback to natural language summary if visualization not applicable.

**Hybrid Questions**  
Decomposes compound questions like "customers from US with orders > $500?" into parallel sub-agents. RAG subgraph + SQL subgraph run simultaneously. Results merged with `synthesise_hybrid` node. Answer synthesis respects document primacy where applicable.

**Query Classification**  
Single LLM call routes every question to document / database / hybrid path. No explicit routing needed from user. Transparent routing logged and displayed.

### Observability & Control

**Agent Trace Logging**  
Every graph node outputs a log entry. UI renders a formatted timeline showing:
- Routing decision (which agent)
- Nodes executed
- Intermediate results
- Final answer

Logs preserve progressive thinking without flooding the user.

**Structured Execution**  
LangGraph `StateGraph[State]` enforces schema. All intermediate values immutable and predictable. Easy to inspect, test, and debug.

**Token Streaming**  
LLM output streamed to UI via `TokenCollector` callback. Tokens burst at superstep boundaries (not continuous, but no threading complexity).

### Deployment & Scale

**Docker Compose Orchestration**  
Separate containers for API (FastAPI) and UI (Streamlit). Docker networking enables local-like development. Compose file scales to AWS EC2.

**AWS EC2 with HTTPS**  
Deployed on t3.small instance. Elastic IP for persistent access. Nginx reverse proxy handles HTTPS via Let's Encrypt. Automatic certificate renewal.

**Rate Limiting**  
Nginx per-IP rate limiting: 20 requests/minute. Prevents abuse without API-level complexity.

### Robustness

**Graceful Error Handling**  
SQL queries auto-retry on error. RAG usefulness loop prevents infinite generation. Escalation node detects unresolved issues (via `issup` and `isuse` signals). Failures logged, never hidden.

**Email Escalation**  
Structured HTML email handoff includes question, conversation history, and support verdict. Gmail SMTP configured via `.env`.

## Engineering Decisions

### 01. What problem does this actually solve?

A chatbot that can answer *any* customer question—whether it lives in documents, the database, or needs both—without the user specifying which. The graph classifies, routes, and executes. The UI shows exactly what happened. It's not just a fallback system; it's an *intelligent* router that proves the answer comes from a real source.

### 02. Why did I build it this way?

| Decision | Why |
|---|---|
| **LangGraph over linear chains** | Needed explicit routing and parallel execution. Hybrid questions decompose and execute RAG + SQL simultaneously. A linear chain cannot fan out and fan in. |
| **Separate API from UI (HTTP)** | Initial architecture had Streamlit importing LangGraph directly. This prevented clean deployment. Refactoring to HTTP boundary between UI and business logic enabled Docker, AWS, and horizontal scaling. |
| **FAISS over managed vector DB** | Zero infrastructure dependencies. Embeddings computed locally (CPU) with `sentence-transformers`. Interface abstracted, so migration to Pinecone/Weaviate is a one-class change. |
| **Direct pymysql over MCP** | Lower latency than an MCP server. TiDB Cloud is MySQL-compatible; pymysql is a direct client. No extra process to manage. |
| **Vega-Lite auto-visualization** | Detects aggregatable SQL results (GROUP BY, COUNT, SUM) and renders charts without manual template work. Gracefully falls back to text if not applicable. |
| **Email over Slack/PagerDuty** | Structured HTML email is portable and human-readable. Escalation decision (via `issup`/`isuse` signals) is explicit and loggable, not implicit. |
| **Docker Compose for local dev** | Same setup locally and on EC2. Single `docker compose up` runs both API and UI with networking pre-configured. No separate container orchestration overhead. |

### 03. What did I consciously trade away?

| Area | What I chose | What I gave up |
|---|---|---|---|
| **Scalability** | In-memory FAISS + local embeddings | Horizontal scaling without shared vector DB |
| **Query complexity** | SQL error recovery loops | Sophisticated semantic SQL generation (e.g., multi-join detection) |
| **Observability** | Structured logs + UI trace | Prometheus/Grafana metrics aggregation |
| **Testing** | Manual validation + live demo | Automated integration tests in CI |
| **Hybrid decomposition** | Simple LLM split into sub-questions | Multi-hop reasoning or iterative refinement |
| **Authentication** | None (internal demo) | API key / JWT for multi-tenant production |
| **Real-time collaboration** | Single conversation per user | Concurrent sessions or shared history |

Every shortcut can be addressed: swap FAISS for Redis, add integration tests, introduce auth. The architecture is designed so no rewrites are necessary.

### 04. What breaks — and how?

| Failure | What happens | What's there now |
|---|---|---|
| **OpenRouter is down** | LLM calls fail; graph returns graceful error to user | Error caught and logged; user sees "I couldn't generate an answer" |
| **TiDB is down** | SQL queries fail; error caught, logged, escalation flagged | `execute_sql` wraps in try-except; `issup` marked as `"partially_supported"` |
| **FAISS index not loaded** | RAG subgraph fails; logs show which node failed | `vector_store.py` loads at startup; health check validates |
| **Email SMTP fails** | Escalation message unsent; logged but continues | Error caught; `handoff_summary` includes reason for human review |
| **Rate limit hit** | 429 from Nginx; request rejected before FastAPI | Nginx per-IP; configurable in reverse proxy |
| **Bad SQL generated** | Query executed, fails, auto-retries with refined SQL | `execute_sql` catches exception, passes error to LLM for refinement |
| **RAG usefulness loop infinite** | Retriever keeps generating; max retries (10) hit | Loop bounded; returns best answer if max retries reached |
| **Large document corpus** | Embeddings take time; startup slow | Mitigated with batch embedding; FAISS index persisted to disk |

### 05. How did I check it works?

Three validation levels:

**Manual Testing (15 scenarios)**  
Documented in `BUILD_LOG.md` and `DEPLOYMENT.md`. Covers:
- Query classification (document / database / hybrid correctly routed)
- SQL execution with error recovery
- RAG retrieval and usefulness loop
- Escalation detection and email handoff
- Rate limiting
- Cache behavior
- Health checks

**Live Demo Validation**  
Deployed to AWS EC2. Public access via HTTPS. Manual testing in production environment confirms:
- End-to-end latency (2-6 seconds typical)
- Concurrent user handling (Nginx rate limiting)
- Database consistency (no stale reads)

**Error Injection**  
Intentional failures tested:
- Invalid SQL (auto-recovery tested)
- Missing documents (RAG fallback tested)
- Rate limit threshold (Nginx behavior validated)

**What's missing:**  
Automated integration tests in pytest. Build validation is manual but comprehensive.

### 06. How does it run beyond my laptop?

**Architecture Refactor**  
Started with Streamlit directly importing LangGraph. Refactored to FastAPI as single execution path, Streamlit as thin HTTP client. This boundary enabled clean deployment.

**Docker Compose**  
`docker-compose.yml` orchestrates two services: API (FastAPI) and UI (Streamlit) with port mappings and environment variables pre-configured.

**AWS EC2 Deployment**  
1. t3.small instance (sufficient for FAISS + sentence-transformers)
2. Elastic IP for persistent public address
3. Security groups allow SSH (my IP) + ports 80/443
4. Nginx reverse proxy from port 80 → Streamlit 8501
5. Let's Encrypt SSL via Certbot
6. Automatic certificate renewal

**To scale horizontally:**
1. Migrate FAISS → Redis vector store (shared across instances)
2. Migrate Streamlit → dedicated frontend container with load balancer
3. Add database connection pooling (pgbouncer / ProxySQL)
4. Replace Nginx → AWS ALB for multi-instance routing
5. Implement session store (Redis / DynamoDB) for conversation history

### 07. What I'd do differently

**Earlier API Boundary**  
The UI/API split should have happened during prototyping, not after. It forced a full refactor mid-project. Lesson: Separate business logic from presentation from day one.

**Automated Testing**  
Manual validation caught issues, but a pytest suite would have been faster. `test_api.py` should have integration tests that run after deployment.

**SQL Agent Robustness**  
Error recovery via LLM refinement works but is slow (2 tokens per retry). A schema-aware query planner would be more reliable.

**Conversation Persistence**  
Currently ephemeral per session. Storing history in DynamoDB / PostgreSQL would enable session recovery and analytics.

**Monitoring Dashboard**  
LangSmith traces are available, but a custom Grafana dashboard showing request latency, error rates, escalation frequency would be useful for operations.

## Architecture

```
┌────────────────────────────────────────────────┐
│  Browser → Nginx (HTTPS ↔ HTTP)                │
├────────────────────────────────────────────────┤
│                                                │
│  ┌─────────────────────────────────────────┐  │
│  │ Streamlit UI :8501 (HTTP client)        │  │
│  │  - Chat interface                       │  │
│  │  - Execute timeline rendering           │  │
│  │  - Vega-Lite visualization             │  │
│  │  - Debug sidebar                        │  │
│  └──────────────────┬──────────────────────┘  │
│                     │                          │
│          requests.post("/chat")                │
│                     │                          │
│  ┌──────────────────▼──────────────────────┐  │
│  │ FastAPI :8000 (Execution layer)         │  │
│  │  ├── /health (ALB ready check)          │  │
│  │  ├── /chat (question + history)         │  │
│  │  └── /docs (Swagger)                    │  │
│  └──────────────────┬──────────────────────┘  │
│                     │                          │
│        ┌────────────▼──────────────┐           │
│        │   LangGraph StateGraph    │           │
│        │  ┌──────────────────────┐ │           │
│        │  │ classify_question    │ │           │
│        │  │  ├→ document         │ │           │
│        │  │  ├→ database         │ │           │
│        │  │  └→ hybrid           │ │           │
│        │  └──────┬─────┬────┬────┘ │           │
│        │         │     │    │      │           │
│        │  ┌──────▼─┐  │    │      │           │
│        │  │RAG Sub │  │    │      │           │
│        │  │ graph  │  │    │      │           │
│        │  └────────┘  │    │      │           │
│        │              │    │      │           │
│        │    ┌──────┐ ┌▼──┐ │      │           │
│        │    │ Sync │ │SQL│ │      │           │
│        │    └──────┘ └───┘ │      │           │
│        │              │    │      │           │
│        │    ┌─────────▼────▼──┐   │           │
│        │    │  synthesise /   │   │           │
│        │    │  finalize       │   │           │
│        │    └─────────────────┘   │           │
│        │              │            │           │
│        │          answer           │           │
│        └────────────┬──────────────┘           │
│                     │                          │
│     ├─ TiDB Cloud (SQL)                        │
│     ├─ FAISS (vector store)                    │
│     ├─ OpenRouter (LLM)                        │
│     └─ Gmail SMTP (escalation)                 │
│                                                │
└────────────────────────────────────────────────┘
```

### Graph Flow: Document Path

```
classify_question
     │
     ▼
"document"
     │
     ▼
retrieve (FAISS search)
     │
     ▼
is_relevant? (LLM check)
     │
     ├─ No → retrieve again (loop up to 10)
     │
     ├─ Yes → generate_from_context
     │
     ▼
check_support (fully / partially / no)
     │
     ▼
check_usefulness (useful / not_useful)
     │
     ├─ not_useful → retrieve again
     │
     ├─ useful → DONE
     │
     ▼
answer
```

### Graph Flow: Database Path

```
classify_question
     │
     ▼
"database"
     │
     ▼
rewrite_sql_query (if vague)
     │
     ▼
generate_sql
     │
     ▼
execute_sql
     │
     ├─ Error → refine query with error, retry
     │
     ├─ Success → visualize_sql (chart spec)
     │
     ▼
summarize_sql (NL answer)
     │
     ▼
db_answer
```

### Graph Flow: Hybrid Path

```
classify_question
     │
     ▼
"hybrid"
     │
     ▼
decompose_question (LLM splits into sub-q)
     │
     ▼
route_sub_questions (Send() fan-out)
     │
     ├─ run_rag_sub ──────┐
     │                    │
     ├─ run_sql_sub ──────┼→ synthesise_hybrid
     │                    │
     └─ ... more subs ────┘
     │
     ▼
hybrid_answer
```

### Failure Handling Strategy

<p align="center">
  <img src="assets/fallback.svg" alt="Multi-Agent Error Recovery Flow" width="400" />
</p>

The application implements multi-layer error handling:

1. **Node-level** — Try-except in every agent node (SQL, RAG, synthesis)
2. **Escalation signals** — `issup` (support verdict) and `isuse` (usefulness) drive escalation detection
3. **Human handoff** — When a question cannot be resolved, escalation node sends structured HTML email
4. **Graceful degradation** — Partial failures are logged; best-effort answer returned to user

## Configuration

All configuration via environment variables. See `.env.example`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenRouter API key |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Primary LLM model |
| `TIDB_HOST` | — | TiDB Cloud host |
| `TIDB_USER` | — | TiDB Cloud user |
| `TIDB_PASSWORD` | — | TiDB Cloud password |
| `TIDB_DATABASE` | `ecommerce_v2` | Database name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `FAISS_INDEX_PATH` | `./faiss_index` | FAISS index location |
| `DOCUMENTS_PATH` | `./documents/` | PDF document directory |
| `GMAIL_SENDER` | — | Gmail address for escalation |
| `GMAIL_PASSWORD` | — | Gmail app password |
| `GMAIL_RECIPIENT` | — | Recipient for escalation emails |
| `MAX_RAG_RETRIES` | `10` | Max retrieval loop retries |
| `RAG_RELEVANCE_THRESHOLD` | `0.5` | Minimum relevance score |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
selfragmcp/
├── app/
│   ├── __init__.py
│   ├── config.py             # Settings, paths, constants
│   ├── state.py              # LangGraph State TypedDict
│   ├── models.py             # Pydantic request/response schemas
│   ├── prompts.py            # RAG + SQL + hybrid LLM prompts
│   ├── llm.py                # get_llm() / get_embeddings()
│   ├── vector_store.py       # FAISS index builder + retriever
│   ├── db_agent.py           # TiDB Cloud connection + queries
│   ├── email_alert.py        # Gmail SMTP escalation handler
│   ├── chat_history.py       # Conversation formatting
│   ├── stream_events.py      # TokenCollector for UI streaming
│   ├── api.py                # FastAPI /health /chat /docs endpoints
│   └── graph/
│       ├── __init__.py
│       ├── builder.py        # StateGraph assembly + compilation
│       ├── nodes.py          # RAG pipeline nodes
│       ├── rag_subgraph.py   # Compiled RAG subgraph (reusable)
│       ├── sql_nodes.py      # SQL agent + hybrid + classifier
│       └── escalation.py     # Escalation detection + email handoff
├── ui/
│   └── streamlit_app.py      # Streamlit UI (HTTP client)
├── documents/                # PDF policy documents
├── main.py                   # CLI entry point
├── Dockerfile                # API container (FastAPI)
├── Dockerfile.ui             # UI container (Streamlit)
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # API dependencies
├── requirements-ui.txt       # UI dependencies
├── .env.example              # Environment template
├── .gitignore
├── README.md                 # This file
├── BUILD_LOG.md              # Build journey (1090 lines)
├── DEPLOYMENT.md             # AWS deployment journey (1148 lines)
└── SYSTEM_DESIGN.md          # Full graph architecture + diagrams
```

## API Reference

### Health Check

```bash
GET /health
```

Response: `{"status": "healthy", "timestamp": "2026-06-13T12:34:56.789Z"}`

### Chat Endpoint

```bash
POST /chat
Content-Type: application/json

{
  "question": "How many customers do we have?",
  "chat_history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"}
  ]
}
```

Response:

```json
{
  "answer": "The total number of customers is 99,441.",
  "query_type": "database",
  "logs": [
    "✅ classify_question: database",
    "✅ rewrite_sql_query: skipped (clear query)",
    "✅ generate_sql: SELECT COUNT(*) FROM customers",
    "✅ execute_sql: 1 row(s)",
    "✅ visualize_sql: aggregatable, chart generated",
    "✅ summarize_sql: converted to natural language"
  ],
  "visualization_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "mark": "bar",
    "data": {"values": [{"metric": "total_customers", "count": 99441}]},
    ...
  },
  "escalated": false,
  "handoff_summary": "",
  "rag_docs_used": null,
  "sql_result": "[{\"COUNT(*)\": 99441}]"
}
```

Full Swagger docs at `/docs`.

## Quick Start

### Clone & Configure

```bash
git clone https://github.com/aniketsarapAI/multi-agent-customer-support
cd selfragmcp
cp .env.example .env
# Fill in .env with API keys
```

### Option A: Docker Compose (Recommended)

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

### Option B: Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# In one terminal:
uvicorn app.api:app --host 0.0.0.0 --port 8000

# In another:
streamlit run ui/streamlit_app.py
```

### Verify

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "healthy", ...}`

## Deployment

### Live Production

**URL:** https://multiagent-ai.localnode.app

**Infrastructure:**
- AWS EC2 t3.small
- Docker Compose (API + UI containers)
- Nginx reverse proxy
- Let's Encrypt SSL/TLS
- Elastic IP (persistent public address)
- Custom DNS (multiagent-ai.localnode.app)

### Deployment Journey

The full 25-phase deployment arc is documented in **[DEPLOYMENT.md](DEPLOYMENT.md)**, covering:

1. Architecture refactor (Streamlit direct → HTTP client)
2. API contract expansion (missing fields)
3. Containerization (Dockerfile, docker-compose)
4. EC2 provisioning (security groups, key pairs)
5. Buildx/BuildKit troubleshooting (version conflicts)
6. Elastic IP + custom domain configuration
7. Nginx reverse proxy setup
8. Let's Encrypt SSL certificate and renewal

Key lessons:
- **UI/API separation is essential for deployment.** Tight coupling prevents scaling.
- **Docker local ≠ Docker AWS.** Buildx and BuildKit versions matter.
- **Start simple: EC2 + Docker Compose.** Premature migration to ECS Fargate adds unnecessary complexity.
- **Always set billing safeguards first.** A $5 monthly budget alert prevents surprises.

### To Deploy Your Own

1. Create AWS account, set $5 budget
2. Launch EC2 t3.small, configure security groups
3. SSH in, clone repo, set `.env`
4. Run `docker compose up -d`
5. Configure Elastic IP + DNS
6. Set up Nginx reverse proxy (template in `DEPLOYMENT.md`)
7. Run Certbot for SSL

Full step-by-step in `DEPLOYMENT.md`.

## Testing & Validation

### Manual Test Scenarios (15 validated)

| Scenario | Query | Expected | Result |
|---|---|---|---|
| RAG retrieval | "What does Bloomly do?" | Document Q&A | ✅ Pass |
| SQL execution | "How many customers?" | Count query + chart | ✅ Pass |
| Hybrid decomposition | "US customers with orders > $500?" | Both agents, merged result | ✅ Pass |
| Query rewriting | "Give me all sales figures" | Vague query refined | ✅ Pass |
| Error recovery | Invalid SQL structure | Auto-refined + retry | ✅ Pass |
| Escalation | "Talk to human" | Email sent, escalated=true | ✅ Pass |
| Conversation | "Thanks!" | Direct NL response | ✅ Pass |
| Rate limiting | 21 requests in 60 sec | 20th succeeds, 21st blocked | ✅ Pass |
| Health check | `curl /health` | 200 OK, healthy status | ✅ Pass |
| Logging | Any query | Logs array populated | ✅ Pass |

**What's missing:**  
Automated pytest integration tests. Validation is comprehensive but manual.

### Live Demo Validation

Deployed to AWS EC2. Public HTTPS access. Manual testing confirms:

- End-to-end latency: 2–6 seconds (typical)
- Concurrent user handling: Nginx rate limiting enforced
- Database consistency: No stale reads
- Email delivery: Escalations arrive in inbox
- Chart rendering: Vega-Lite specs render correctly in Streamlit

### Error Injection Testing

Intentional failures tested in production:
- Invalid SQL → auto-recovery verified
- Missing document → RAG fallback behavior correct
- Rate limit threshold → Nginx behavior enforced
- FAISS index missing → Startup validation caught

## Roadmap

Planned improvements:

### Phase 2: Robustness

- [ ] Pytest integration tests (currently manual)
- [ ] Prometheus metrics (replace in-memory counters)
- [ ] LangSmith tracing integration
- [ ] Redis vector store migration (from in-memory FAISS)
- [ ] Session persistence (conversation history in DynamoDB)

### Phase 3: Scale

- [ ] Multi-instance deployment (ALB + ECS Fargate)
- [ ] Database connection pooling (ProxySQL)
- [ ] API authentication (JWT / API keys)
- [ ] Custom Grafana dashboard (request latency, error rates, escalation frequency)
- [ ] Async worker queue for long-running tasks

### Phase 4: Intelligence

- [ ] Multi-turn context refinement (iterate on vague questions)
- [ ] Schema-aware SQL planner (reduce error loops)
- [ ] Semantic SQL generation (multi-join queries)
- [ ] Document clustering (relevant document prioritization)
- [ ] Feedback loop (user upvotes/downvotes for reranking)

---

**Built by Aniket Sarap**  
**Production deployment:** June 2026  
**Last updated:** June 13, 2026