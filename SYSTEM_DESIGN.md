# System Design

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    app/api.py (FastAPI)                   │
│  POST /chat → Application.supervisor.stream(...)          │
│  config={"configurable": {"thread_id": conversation_id}}  │
│  SqliteSaver checkpointer for multi-turn continuity      │
└──────────────────────────┬───────────────────────────────┘
                           │ question + chat_history
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Planner (app/orchestration/planner.py)                  │
│  1 LLM call: classify → query_type                       │
│  document / database / hybrid / conversation             │
│  → ExecutionPlan(agents, needs_synthesis, cacheable)     │
└──────────────────────────┬───────────────────────────────┘
                           │ ExecutionPlan
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Supervisor Graph (StateGraph[SupervisorState])          │
│  Checkpointer: SqliteSaver (thread_id = conversation_id) │
│                                                          │
│  ┌──────────────┐                                        │
│  │ decompose    │  hybrid only: LLM splits question      │
│  └──┬───┬───┬──┘                                         │
│     │   │   │                                             │
│     │   │   └──────────────┐                              │
│     │   │                  │ (no decomposition)           │
│  ┌──▼──┐   ┌──────────┐   ┌▼──────────┐                  │
│  │Send │   │ Send     │   │ reclassify│  2nd LLM call    │
│  │×N   │   │ × agents │   └──┬───┬────┘                  │
│  └──┬──┘   └────┬─────┘      │   │                       │
│     │           │            │   │                       │
│  ┌──▼───────────▼────────────▼───┐                       │
│  │  rag_agent / sql_agent /      │  parallel via Send    │
│  │  conversation_agent           │                       │
│  │  (try/except → failed result) │                       │
│  └──────────────┬────────────────┘                       │
│                 │ agent_results (operator.add reducer)   │
│          ┌──────▼──────┐                                 │
│          │ synthesise  │  hybrid: 1 LLM call             │
│          │             │  single: pass-through           │
│          └──────┬──────┘                                 │
│                 │                                        │
│          ┌──────▼──────┐                                 │
│          │ escalate    │  EscalationChecker (fire-and-   │
│          │             │  forget email + handoff summary)│
│          └──────┬──────┘                                 │
│                 │                                        │
│                END                                       │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Output: CLI / Streamlit UI                              │
│  - Final answer text                                     │
│  - Logs (progressive thinking)                           │
│  - Vega-Lite chart (if SQL result)                       │
│  - Debug info (query_type, issup, isuse, agent_count)    │
│  - Escalation info (reason, handoff_summary)             │
└──────────────────────────────────────────────────────────┘
```

## Agent Subgraphs

### RAG Agent (app/agents/rag_agent.py → app/agents/graph/rag_subgraph.py)

```
START → decide_retrieval
         │
         ├─(no retrieval)→ generate_direct → END
         │
         └─(retrieve)→ retrieve → is_relevant
                                        │
                   ┌─(relevant docs)─────┘
                   ▼
         generate_from_context → is_sup
                                   │
                   ├─(fully_supported)→ is_use
                   │                      │
                   │                      ├─(useful)→ END
                   │                      └─(not_useful)→ rewrite_question → retrieve
                   │
                   └─(not supported)→ revise_answer → is_sup  (max 5 retries)
```

- Self-RAG pattern: IsSUP (grounding check) → Revise loop, IsUSE (usefulness check) → Rewrite loop
- `MAX_RETRIES = 5` (is_sup/revise), `MAX_REWRITE_TRIES = 3` (is_use/rewrite)
- Case-insensitive comparison in routers (`"fully_supported"`, `"useful"`)
- Structured outputs with safe defaults on LLM failure

### SQL Agent (app/agents/sql_agent.py → app/agents/graph/sql_nodes.py)

```
START → rewrite_sql_query → generate_sql → execute_sql
                                               │
                    ┌──────────────────────────┘
                    │
              ┌─(success)──→ visualize_sql → summarize_sql → END
              │
              ├─(syntax error, retry_count < 3)──→ generate_sql (retry with error context)
              │
              ├─(validation error)──→ visualize_sql (with db_error) → END
              │
              └─(connection error)──→ visualize_sql (with db_error) → END
```

- `SELECT`-only validation prevents DDL/DML
- Error classification: `SQLValidationError`, `SQLSyntaxError` (retryable), `SQLConnectionError` (not retryable)
- `original_question` preserved for summarizer (user's literal question, not refined query)
- Retry as graph edges (not hidden Python loop) — visible in graph trace

### Conversation Agent (app/agents/conversation_agent.py)

Single LLM call, no subgraph. Handles greetings, emotions, small talk.

## State Fields

### SupervisorState
| Field | Type | Reducer | Purpose |
|---|---|---|---|
| `request_id` | `str` | — | Request tracking |
| `conversation_id` | `str` | — | Used as `thread_id` for checkpointer |
| `question` | `str` | — | Current question (may be overridden per sub-question) |
| `original_question` | `str` | — | User's literal input |
| `chat_history` | `list[dict]` | — | Conversation context |
| `execution_plan` | `ExecutionPlan?` | — | Agent routing plan |
| `sub_questions` | `list[dict]` | — | Decomposed sub-questions |
| `reclassified_query_type` | `str` | — | Fallback classification for non-decomposable hybrid |
| `agent_results` | `list[AgentResult]` | `operator.add` | Accumulates parallel agent results |
| `final_answer` | `str` | — | Synthesized answer |
| `logs` | `list[str]` | `operator.add` | Progressive thinking log |
| `visualization_spec` | `dict?` | — | Vega-Lite chart spec |
| `escalated` | `bool` | — | Whether escalation fired |
| `escalation_reason` | `str` | — | Escalation reason |
| `handoff_summary` | `str` | — | Human handoff summary |

## Key Design Decisions

- **LangGraph checkpointer (SqliteSaver)** — Multi-turn state continuity via `thread_id`; no manual Redis memory round-trip
- **`Send` for parallel fan-out** — One `Send` per agent or per decomposed sub-question; `agent_results` uses `operator.add` reducer for fan-in
- **Hybrid re-classification** — If decomposition yields no sub-questions, a second LLM classification call picks a single dominant agent
- **Agent error containment** — `try/except` around `agent.invoke`; failed agents return `AgentResult(success=False)`; synthesise works with remaining successes
- **Escalation as graph node** — `escalate` node runs after `synthesise`; fire-and-forget email + handoff summary
- **SQL retry as graph edges** — `execute_sql →(conditional)→ generate_sql` cycle; retry visible in graph trace
- **`SELECT`-only validation** — Rejects non-SELECT/WITH and multi-statement SQL before execution
- **Error classification** — `SQLSyntaxError` (retryable) vs `SQLConnectionError` (not retryable) vs `SQLValidationError` (not retryable)
- **Structured outputs with safe defaults** — Every `with_structured_output` call passes a deterministic default factory; double-LLM-failure returns the default instead of crashing
- **Case-insensitive routers** — `issup`/`isuse` comparisons use `.lower()` to handle LLM casing variations
- **FAISS index persistence** — Index saved to `faiss_index/` on first build, loaded on subsequent startups
- **Burst token streaming** — (Not yet wired; `TokenCollector` removed as dead code)

## Project Layout

```
selfragmcp/
├── app/
│   ├── __init__.py
│   ├── api.py                     — FastAPI server entry point
│   ├── config.py                  — Settings (model, retriever params, DB)
│   ├── prompts.py                 — All prompt templates
│   ├── chat_history.py            — format_chat_history helper
│   ├── graph/
│   │   └── builder.py             — Application container (wires everything)
│   ├── models/
│   │   ├── state.py               — TypedDicts (SupervisorState, RAGAgentState, etc.)
│   │   ├── agent.py               — BaseAgent protocol, AgentResult
│   │   ├── planning.py            — ExecutionPlan, QueryTypeDecision, etc.
│   │   └── metadata.py            — RAGMetadata, SQLMetadata, decision schemas
│   ├── orchestration/
│   │   ├── planner.py             — Planner (classify → ExecutionPlan)
│   │   ├── supervisor.py          — build_supervisor_graph
│   │   └── registry.py            — AgentRegistry
│   ├── agents/
│   │   ├── rag_agent.py           — RAGAgent (wraps RAG subgraph)
│   │   ├── sql_agent.py           — SQLAgent (wraps SQL subgraph)
│   │   ├── conversation_agent.py  — ConversationAgent (single LLM call)
│   │   └── graph/
│   │       ├── nodes.py           — RAG node factories + routers
│   │       ├── rag_subgraph.py    — build_rag_subgraph
│   │       └── sql_nodes.py       — SQL node factories + build_sql_subgraph
│   ├── infrastructure/
│   │   ├── llm.py                 — LLMWithFallback, get_llm, get_embeddings
│   │   ├── vector_store.py        — FAISS index (persisted) + retriever
│   │   ├── db_agent.py            — pymysql client, SQL validation, error types
│   │   └── email_alert.py         — SMTP escalation email
│   ├── pipeline/
│   │   ├── escalation.py          — EscalationChecker
│   │   └── post_processing.py     — PostProcessingPipeline (security output check)
│   └── services/
│       ├── cache.py               — SemanticCache (FAISS + Redis)
│       ├── memory.py              — RedisMemoryService / InMemoryMemoryService
│       ├── security.py            — InputSanitizer, PIIDetector, OutputValidator
│       ├── health.py              — HealthService
│       └── monitoring.py          — MetricsCollector, JSONFormatter
├── ui/
│   └── streamlit_app.py           — Streamlit UI (HTTP client)
├── documents/                     — PDF policy docs
├── tests/                         — pytest tests
└── faiss_index/                   — Persisted FAISS index (gitignored)
```
