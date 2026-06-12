# System Design

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                         main.py / streamlit_app.py       │
│  ┌─────────────────────────────────────────────────┐     │
│  │  app.stream(stream_mode="values")               │     │
│  │  + TokenCollector callback for LLM token streaming│    │
│  └──────┬──────────────────────────────────────────┘     │
└─────────┼────────────────────────────────────────────────┘
          │ question
          ▼
┌──────────────────────────────────────────────────────────┐
│  Unified LangGraph (StateGraph[State])                   │
│                                                          │
│  ┌──────────────┐                                        │
│  │ classifY     │  router: document / database / hybrid  │
│  └──┬───┬───┬──┘                                         │
│     │   │   │                                             │
│     │   │   └──────────────────┐                          │
│     │   │                      │                          │
│  ┌──▼──┐   ┌──────────────┐   ┌▼──────────┐              │
│  │ RAG │   │ SQL Path     │   │ Decompose  │  hybrid      │
│  │ sub-│   │              │   └──┬───┬─────┘              │
│  │graph│   │ rewrite_sql  │      │   │                    │
│  │     │   │ generate_sql │   ┌──▼───▼──┐                 │
│  │     │   │ execute_sql  │   │ Send × N │  parallel      │
│  │     │   │ visualize_   │   └──┬───┬───┘                 │
│  │     │   │   sql        │      │   │                     │
│  │     │   │ summarize_   │   ┌──▼───▼──┐                 │
│  │     │   │   sql        │   │ run_rag_ │  run_sql_sub    │
│  └──────┘   └──────────────┘   │ sub      │                │
│       │            │           └─────┬─────┘                │
│       └──────┬─────┘                 │                      │
│              │                 ┌──────▼──────┐              │
│              └─────────────────►  synthesise  │  fan-in     │
│                                └──────┬──────┘              │
│                                       │                     │
│                                    END│                     │
│                                       │                     │
│  Node outputs include "logs" entries  │                     │
│  and "visualization_spec" for charts  │                     │
└──────────────────────────────────────────────────────────┘
          │ answer (or db_answer)
          ▼
┌──────────────────────────────────────────────────────────┐
│  Output                    CLI          Streamlit         │
│  ┌──────────────────────────────────────────────────┐     │
│  │ - Final answer text    print( )    st.markdown( )│     │
│  │ - Logs (12 latest)     print( )    sidebar       │     │
│  │ - Vega-Lite chart      N/A         st.vega_lite  │     │
│  │ - Debug info           print( )    st.expander   │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

## Graph Flows

### Document Path
1. `classify_question` → `"retrieve"` → `rag_pipeline` (compiled subgraph)
2. Subgraph: `decide_retrieval` → `retrieve` → `is_relevant` (×N) → `generate_from_context` → `check_support` → `check_usefulness` (loops if not useful, up to 10 retries)

### Database Path
1. `classify_question` → `"rewrite_sql_query"` → `rewrite_sql_query` (optional refinement for vague queries)
2. `generate_sql` → `execute_sql` (with auto-retry on error) → `visualize_sql` (chart spec) → `summarize_sql` (NL answer)

### Hybrid Path
1. `classify_question` → `"hybrid"` → `decompose_question` (LLM splits into sub-questions)
2. `route_sub_questions` uses `Send()` to fan out to `run_rag_sub` / `run_sql_sub` in parallel
3. `synthesise_hybrid` merges all sub-results into a single answer

## State Fields

| Field | Type | Purpose |
|---|---|---|
| `question` | `str` | Input question |
| `query_type` | `"document" | "database" | "hybrid"` | Routed type |
| `sub_questions` | `List[dict]` | Decomposed sub-questions |
| `sub_results` | `Sequence[tuple[str, str]]` | (id, answer) pairs from parallel execution |
| `visualization_spec` | `dict | None` | Vega-Lite bar chart spec |
| `sql_query` | `str` | Generated SQL |
| `sql_result` | `str` | JSON result rows |
| `db_answer` | `str` | NL answer from SQL |
| `answer` | `str` | NL answer from RAG |
| `logs` | `list[str]` | Progressive thinking log entries |
| `docs` | `List[Document]` | Retrieved documents |
| `relevant_docs` | `List[Document]` | Filtered relevant docs |
| `issup` | `"fully_supported" | "partially_supported" | "no_support"` | Support verdict |
| `isuse` | `"useful" | "not_useful"` | Usefulness verdict |

## Project Layout

```
selfragmcp/
├── app/
│   ├── __init__.py
│   ├── config.py              — Settings (model, retriever params, DB)
│   ├── state.py               — State TypedDict
│   ├── models.py              — Pydantic models for structured output
│   ├── prompts.py             — RAG prompt templates
│   ├── llm.py                 — get_llm() / get_embeddings()
│   ├── stream_events.py       — TokenCollector callback
│   ├── vector_store.py        — FAISS index builder + retriever
│   ├── db_agent.py            — Direct pymysql client for TiDB
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py         — Graph assembly + compilation
│   │   ├── nodes.py           — RAG pipeline nodes
│   │   ├── rag_subgraph.py    — Compiled RAG subgraph for reuse
│   │   └── sql_nodes.py       — SQL + hybrid + classify nodes
│   └── prompts.py             — RAG + SQL + hybrid prompts
├── ui/
│   └── streamlit_app.py       — Streamlit UI
├── documents/                 — PDF policy docs
├── main.py                    — CLI entry point
├── README.md
└── SYSTEM_DESIGN.md
```

## Key Design Decisions

- **Direct pymysql over MCP** — Lower latency, no npm dependency, no standalone server process
- **Compiled RAG subgraph** — Enables clean parallel execution for hybrid questions without duplicating node logic
- **`Send` for hybrid fan-out** — LangGraph's `Send` enables parallel sub-graph execution with automatic fan-in via `operator.add`
- **Burst token streaming** — Tokens arrive at superstep boundaries (not continuous), but threading-free and simple
- **Error recovery in SQL** — Failed queries auto-retry with refined SQL using the error message as context
