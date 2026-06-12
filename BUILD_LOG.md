# Build Log: Multi-Agent Customer Support with Self-RAG, SQL & Human Escalation

> A first-person account of building a LangGraph-based customer support system — what I learned, what I'd do differently, and how to build one yourself.

---

## Table of Contents

1. [How this project started](#1-how-this-project-started)
2. [Planning — what to plan before writing code](#2-planning--what-to-plan-before-writing-code)
3. [State first — design what flows through the graph](#3-state-first--design-what-flows-through-the-graph)
4. [Models — typed contracts for every LLM call](#4-models--typed-contracts-for-every-llm-call)
5. [Prompts — the real source of behavior](#5-prompts--the-real-source-of-behavior)
6. [Nodes — pure functions on state](#6-nodes--pure-functions-on-state)
7. [Graph assembly — connecting the pieces](#7-graph-assembly--connecting-the-pieces)
8. [The RAG subgraph — Self-RAG in practice](#8-the-rag-subgraph--self-rag-in-practice)
9. [The SQL path — where most of the bugs lived](#9-the-sql-path--where-most-of-the-bugs-lived)
10. [Hybrid path — parallel RAG + SQL](#10-hybrid-path--parallel-rag--sql)
11. [Conversation path — routing emotional messages](#11-conversation-path--routing-emotional-messages)
12. [Escalation — the hardest part to tune](#12-escalation--the-hardest-part-to-tune)
13. [Email — deterministic HTML alerts](#13-email--deterministic-html-alerts)
14. [What I learned the hard way](#14-what-i-learned-the-hard-way)
15. [What I'd keep the same](#15-what-id-keep-the-same)
16. [What I'd change](#16-what-id-change)
17. [Production considerations](#17-production-considerations)
18. [Checklist for building similar systems](#18-checklist-for-building-similar-systems)
19. [The short version — TL;DR](#19-the-short-version--tldr)

---

## 1. How this project started

I had a Jupyter notebook implementing **Self-RAG** — the paper from IBM/Meta where a generator produces an answer, a verifier checks if it's supported by retrieved context (IsSUP), and if not, a reviser rewrites it using only quotes from the context, then a usefulness check (IsUSE) decides if the loop continues.

That notebook worked for one thing: answering questions from a set of PDF documents. It had:

- A retrieval step (FAISS over company policy PDFs)
- A relevance filter
- A generation step
- A support verification step (IsSUP)
- A revision loop when support was weak
- A usefulness check (IsUSE)
- A rewrite step when usefulness was low

It was a great research paper implementation. But it was **one query type, one path, one notebook**.

The project grew from there — I wanted to answer questions about e-commerce data (customers, orders, revenue) stored in a TiDB Cloud database too. Then I wanted compound questions that needed both documents and database queries. Then I needed conversation routing for emotional messages. Then escalation for when the system couldn't help.

The notebook became a LangGraph, and that became the project you see here.

### The original notebook structure

```
User question
  ↓
Decide if retrieval is needed
  ↓ (if yes)
Retrieve from FAISS
  ↓
Filter by relevance
  ↓
Generate answer from context
  ↓
IsSUP — is the answer supported by the context?
  ├── fully_supported → continue
  └── partially_supported / no_support → revise answer (loop)
        ↓
IsUSE — is the answer useful?
  ├── useful → done
  └── not_useful → rewrite question and retry (loop)
```

### What I wanted it to become

```
User question
  ↓
Classify: document, database, hybrid, or conversation?
  ├── document → RAG (the original Self-RAG loop)
  ├── database → NL → SQL → execute → visualize → summarize
  ├── hybrid → decompose → parallel RAG + SQL → merge
  └── conversation → direct response (no retrieval)
        ↓
Check: does this need human escalation?
        ↓
Generate handoff or END
```

---

## 2. Planning — what to plan before writing code

### The biggest mistake I made

I started coding nodes before designing the state.

This meant I kept adding fields to the state TypedDict as I went — sometimes fields I never used (`conversation_summary` sat in the state for weeks before I noticed). It also meant my state ended up with 28 fields, some of which are only relevant to one specific path.

### The right order

```
1. State      → What data flows through the system?
2. Models     → What structured outputs does the LLM produce?
3. Prompts    → What instructions does each LLM call need?
4. Nodes      → What does each step do to the state?
5. Graph      → How do the nodes connect?
6. Builder    → How do we compile and expose the graph?
```

### What to plan

| Question | Why it matters |
|----------|----------------|
| What are the input types? | Question text, chat history — these define the entry point |
| What are the query types? | Document, database, hybrid, conversation — these define the routing |
| What does each path produce? | Answer, db_answer, sql_query, visualization_spec — these define state fields |
| What crosses all paths? | Logs, escalation signals — these define shared state |
| Where do paths converge? | Escalation check — this defines the graph structure |

### Decision tree for routing

```
What kind of question is this?
├── About company policies, HR, pricing, contracts?
│   → document → RAG path
├── About e-commerce data, sales, revenue, customers?
│   → database → SQL path
├── Has multiple parts spanning both?
│   → hybrid → decompose and parallel execute
└── Emotional, feedback, conversational?
    → conversation → direct answer, skip retrieval
```

### The planning artifact I wish I'd created

A simple table of every state field with:

| Field | Set by which node(s)? | Read by which node(s)? | Path-specific? |
|-------|----------------------|----------------------|----------------|
| question | Input | classify, retrieve, generate_sql, etc. | No |
| query_type | classify | route_after_classify | No |
| answer | generate_from_context, generate_direct, synthesise_hybrid | is_sup, is_use, check_escalation | No |
| sql_query | generate_sql, run_sql_sub | execute_sql_node, visualize_sql | Yes (database/hybrid) |
| ... | ... | ... | ... |

I didn't make this table. I should have. It would have caught `conversation_summary` before it ever hit the state.

---

## 3. State first — design what flows through the graph

The state is a `TypedDict` — it defines every field that any node can read or write. LangGraph passes this dict between nodes. Each node returns a partial dict that gets merged into the state.

### The final state (cleaned up)

```python
class State(TypedDict):
    # Input
    question: str
    original_question: str
    query_type: Literal["document", "database", "hybrid", "conversation"]

    # Conversation
    chat_history: Annotated[list[dict], operator.add]

    # Hybrid path
    sub_questions: List[dict]
    sub_results: Annotated[Sequence[tuple[str, str]], operator.add]
    sub_question: Optional[dict]

    # RAG path
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str

    # All paths
    logs: Annotated[list[str], operator.add]

    # SQL path
    sql_query: str
    sql_result: str
    db_answer: str
    db_error: str
    visualization_spec: Optional[dict]

    # Escalation path
    escalated: bool
    escalation_reason: str
    handoff_summary: str

    # Accumulated metadata (for email)
    rag_docs_used: list[str]
    sql_queries_executed: list[str]
```

### Key design decisions in the state

**`Annotated[..., operator.add]`** — Fields with `operator.add` accumulate across nodes instead of overwriting. This is how `sub_results` from parallel hybrid nodes get merged and how `logs` accumulate across the graph without losing entries.

**`query_type` is a `Literal`** — The classifier sets it once, and the router reads it once. Using a Literal type means Python/my editor catches typos like `"datbase"`.

**Optional fields get defaults** — `sub_question`, `visualization_spec` default to `None`. All string fields default to `""`, bools to `False`, lists to `[]`.

### What I'd keep

- Single TypedDict for the whole graph — no per-path substates
- `operator.add` for logs and sub_results
- Literal types for enum-like fields

### What I'd change

I'd remove `conversation_summary` — it was defined early, initialized to `""`, and never read by any node. If a field doesn't appear in at least one node's `return` dict, it doesn't belong in the state.

---

## 4. Models — typed contracts for every LLM call

Every structured LLM output in the project is defined as a Pydantic model. LangChain's `with_structured_output()` uses the model to parse the LLM's JSON response into a typed Python object.

### Why this matters

Without a model, the LLM might return:

```json
{"query_type": "database"}
```

or:

```json
{"type": "database"}
```

or:

```json
The answer is database.
```

With a model, LangChain tries to parse the response into the expected schema. If parsing fails, the LLM gets retried with the error message. I rarely see malformed responses in practice because of this.

### The models I ended up with

```python
class EscalationDecision(BaseModel):
    escalate: bool
    reason: Literal["human_requested", "complaint", "frustration",
                     "repeated_negative_sentiment", "unresolved_issue", "none"]

class QueryTypeDecision(BaseModel):
    query_type: Literal["document", "database", "hybrid", "conversation"]

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str

class SQLQueryDecision(BaseModel):
    sql_query: str

class DecomposeDecision(BaseModel):
    sub_questions: List[SubQuestionItem]
```

### What I learned

- **One model per structured LLM call** — don't reuse models across different prompts, even if they look similar. The field descriptions (via `Field(description=...)`) should be specific to each use case.
- **Literal types prevent drift** — `issup` can only be one of three strings. If the LLM returns something else, it gets retried.
- **Keep models simple** — no nested logic, no validators, no computed fields. They're data contracts, not business logic.

---

## 5. Prompts — the real source of behavior

The prompts are the most important files in the project. They define how the LLM behaves at every step. I spent more time tuning prompts than any other file.

### Prompt organization

All prompts live in `app/prompts.py`. Each is a `ChatPromptTemplate` with system instructions + human message template.

### The classifier prompt (most critical prompt)

```python
classify_question_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's question into one of four types:
- "document": Company policies, HR, pricing, etc.
- "database": E-commerce data, sales, revenue, etc.
- "hybrid": Multiple parts spanning both types
- "conversation": Emotion, feedback, conversational intent

Examples:
  'What is the refund policy?' -> document
  'How many customers do we have?' -> database
  'What is the refund policy and how many customers?' -> hybrid
  'This is frustrating' -> conversation

If unsure or the question is general, return 'document'.
Use conversation history when relevant."""),
    ("human", "Conversation History:\n{chat_history}\n\nCurrent Question:\n{question}"),
])
```

### What I learned about prompt design

**Be concrete with examples** — The classifier prompt includes 15+ examples covering edge cases (e.g., "Top selling items in beauty, watches, and bedding" → hybrid because it's 3 independent SQL queries). Without these examples, the classifier frequently misclassifies.

**Separate "what" from "how"** — System prompts say what to do. Human messages provide the data. This separation makes prompts reusable and testable.

**Repeat critical rules in every prompt that needs them** — The SQL generation prompt has a 30-line section about Portuguese category names. The SQL retry prompt has the same section. This violates DRY but ensures the LLM always gets the context it needs, regardless of which prompt path it took.

**The escalation prompt was the hardest to get right** — See section 12.

### The SQL generation prompt (most complex prompt)

This prompt includes:
- The full table schema (11 tables, ~60 columns)
- Rules about SELECT-only, LIMIT, column verification
- Portuguese category name handling instructions
- The "never invent filter values" rule

The SQL prompt is 40 lines. The table schema is passed as a variable, not hardcoded.

### The template injection pattern that worked well

```python
# In the prompt:
"Available tables:\n{table_schema}\n\n"

# In the node:
generate_sql_prompt.format_messages(
    question=question,
    table_schema=TABLE_SCHEMA,
    chat_history=chat_history,
)
```

This keeps the schema out of the prompt file and in the DB client where it belongs.

---

## 6. Nodes — pure functions on state

Every node in the graph is a function that receives the current `State` dict and returns a partial dict of updates. LangGraph merges these updates into the state for the next node.

### A basic node

```python
def classify_question(state: State):
    logs = ["🔍 classify_question: classifying question type..."]
    chat_history = format_chat_history(state.get("chat_history", []))
    structured = _llm.with_structured_output(QueryTypeDecision)
    decision: QueryTypeDecision = structured.invoke(
        classify_question_prompt_v2.format_messages(
            question=state["question"],
            chat_history=chat_history,
        )
    )
    logs.append(f"✅ classify_question: {decision.query_type}")
    return {"query_type": decision.query_type, "logs": logs}
```

### Node conventions I followed

**Log everything** — Every node appends to `logs`. This is the only way to see what the graph is doing during execution. The last 12 logs are shown in the Streamlit UI.

**Read conservatively, write precisely** — Nodes use `state.get("field", default)` to avoid KeyError. They return only the fields they actually change.

**One LLM call per node** — If a node needs multiple LLM calls, it probably should be multiple nodes. The exception is `is_relevant` which calls the LLM once per document — but that's the same call in a loop, not different calls.

**Set quality signals on every answer** — Every node that produces an answer also sets `issup` and `isuse`. This is critical for the escalation check (section 12).

### The three kinds of nodes

| Kind | Example | Purpose |
|------|---------|---------|
| **Processing** | `retrieve`, `execute_sql_node` | Do work, no LLM call |
| **LLM** | `classify_question`, `generate_sql` | Call the LLM, parse structured output |
| **Routing** | `route_after_classify`, `route_escalation` | Return a string to determine which edge to follow |

### The pattern I use for LLM nodes

```python
def some_llm_node(state: State):
    logs = ["🔍 some_llm_node: doing something..."]
    # Prepare inputs
    context = format_chat_history(state.get("chat_history", []))
    # Call LLM with structured output
    structured = _llm.with_structured_output(SomeDecision)
    decision: SomeDecision = structured.invoke(
        some_prompt.format_messages(
            question=state["question"],
            context=context,
        )
    )
    logs.append(f"✅ some_llm_node: done — {decision.some_field}")
    # Return updates
    return {"some_field": decision.some_field, "logs": logs}
```

---

## 7. Graph assembly — connecting the pieces

The graph is built using `StateGraph` from `langgraph.graph`. Nodes are added with `add_node()`, edges with `add_edge()` or `add_conditional_edges()`.

### The main entry point

```python
g = StateGraph(State)

g.add_node("classify_question", classify_question)

# RAG path
g.add_node("rag_pipeline", rag_subgraph)

# SQL path
g.add_node("rewrite_sql_query", rewrite_sql_query)
g.add_node("generate_sql", generate_sql)
g.add_node("execute_sql", execute_sql_node)
g.add_node("visualize_sql", visualize_sql_result)
g.add_node("summarize_sql", summarize_sql_result)

# Hybrid path
g.add_node("decompose_question", decompose_question)
g.add_node("run_rag_sub", run_rag_sub)
g.add_node("run_sql_sub", run_sql_sub)
g.add_node("synthesise_hybrid", synthesise_hybrid)

# Conversation path
g.add_node("generate_direct", generate_direct)

# Escalation
g.add_node("check_escalation", check_escalation)
g.add_node("generate_handoff", generate_handoff)

g.add_edge(START, "classify_question")
```

### Conditional routing

```python
g.add_conditional_edges(
    "classify_question",
    route_after_classify,
    {
        "retrieve": "rag_pipeline",
        "rewrite_sql_query": "rewrite_sql_query",
        "decompose_question": "decompose_question",
        "generate_direct": "generate_direct",
    },
)
```

The `route_after_classify` function reads `query_type` from state and returns the matching key:

```python
def route_after_classify(state):
    qt = state.get("query_type", "document")
    if qt == "database":
        return "rewrite_sql_query"
    elif qt == "hybrid":
        return "decompose_question"
    elif qt == "conversation":
        return "generate_direct"
    else:
        return "retrieve"
```

### Convergence pattern

All four paths converge at `check_escalation`:

```python
g.add_edge("rag_pipeline", "check_escalation")
g.add_edge("generate_direct", "check_escalation")
g.add_edge("summarize_sql", "check_escalation")
g.add_edge("synthesise_hybrid", "check_escalation")
```

This is the key architectural insight: **classify diverges, escalation converges**. Every path leads to the same escalation check, which then decides if the conversation continues or hands off to a human.

### The graph flow (visualized)

```
START → classify
         ↓
    ┌────┼────┬──────┐
    │    │    │      │
  RAG  SQL Hybrid Conv
    │    │    │      │
    └────┼────┘      │
         ↓           │
      escalate ◄─────┘
       ↓     ↓
    handoff  END
```

---

## 8. The RAG subgraph — Self-RAG in practice

The RAG path is the most complex. It's a compiled subgraph with 10 internal nodes that implement the Self-RAG loop.

### Why a compiled subgraph

The RAG path is reused in two places:

1. **Main document queries** — User asks a policy question
2. **Hybrid sub-queries** — A decomposed hybrid question has one or more `doc_*` sub-questions

A compiled subgraph (`rag_subgraph`) can be invoked from the main graph or passed to `run_rag_sub` in the hybrid path. Without it, I'd need to duplicate 10 nodes.

### The self-rag loop

```
START
  ↓
decide_retrieval (should I retrieve or answer directly?)
  ↓ (retrieve)
retrieve (FAISS search)
  ↓
is_relevant (filter documents by topic relevance)
  ↓ (relevant docs exist)
generate_from_context (write answer using context)
  ↓
is_sup (is the answer supported by context?)
  ├── fully_supported → is_use
  └── partially_supported / no_support → revise_answer → is_sup (loop)
        ↓
is_use (is the answer useful?)
  ├── useful → END
  └── not_useful → rewrite_question → retrieve (loop)
```

### What each node does

| Node | LLM call? | Reads | Writes |
|------|-----------|-------|--------|
| decide_retrieval | Yes | question | need_retrieval |
| retrieve | No | retrieval_query | docs (FAISS results) |
| is_relevant | Yes (×N docs) | question, docs | relevant_docs |
| generate_from_context | Yes | question, context | answer |
| is_sup | Yes | question, answer, context | issup, evidence |
| revise_answer | Yes | question, answer, context | answer (quotes-only) |
| is_use | Yes | question, answer | isuse, use_reason |
| rewrite_question | Yes | question, history | retrieval_query |

### The IsSUP loop in detail

The support check is the most important part of Self-RAG:

1. Generate an answer from context
2. Check: is every claim in the answer supported by the context?
3. If fully supported, proceed to usefulness check
4. If partially supported or no support, revise the answer (replace it with direct quotes from context)
5. Repeat up to `MAX_RETRIES` (10) times

The revision step is strict — the LLM outputs only bullet points with direct quotes, no new text:

```
- "RAG-gator offers a 30-day money-back guarantee on all annual plans"
- "Refund must be requested within 14 days of purchase"
```

### What I learned about the Self-RAG loop

**The loop rarely runs more than 1-2 iterations** — The first answer is usually well-supported by the context. The revision step exists for edge cases where the LLM hallucinates.

**The rewrite step is the most expensive** — It re-runs retrieval, relevance filtering, generation, support check, AND usefulness check. Avoid it by having a good classifier and retrieval query up front.

**IsSUP + IsUSE together prevent most hallucinated answers** — Not all, but most. The combination catches answers that are grounded but useless (the document has the right topic but the answer is generic) and answers that are useful but fabricated.

---

## 9. The SQL path — where most of the bugs lived

The SQL path converts natural language questions into MySQL queries against the TiDB Cloud `ecommerce_v2` database (11 tables, 1.5M rows).

### The flow

```
rewrite_sql_query (refine vague queries)
  ↓
generate_sql (NL → MySQL)
  ↓
execute_sql_node (run query, retry on error up to 3x)
  ↓
visualize_sql_result (Vega-Lite chart if numeric+cat columns)
  ↓
summarize_sql_result (NL summary of results)
```

### Why the SQL path was the buggiest

**Column hallucination** — The LLM frequently invents column names that don't exist in the schema. Example: `product_name` when the actual column is `product_category_name`. Fix: aggressive instruction in the prompt to verify every column name against the schema block.

**Portuguese category names** — The `products` table stores categories in Portuguese (`beleza_saude`, `relogios_presentes`). A `product_category_name_translation` table maps them to English (`health_beauty`, `watches_gifts`). The prompt must tell the LLM about both the translation table and the exact format (underscore-separated, lowercase).

**Invented filter values** — The LLM loves to create realistic-looking but fake IDs: `WHERE product_id = 'ABC123'`. The "never invent filter values" rule took several prompt iterations to stick.

**SQL retry loop** — When a query fails, the error message is fed back to the LLM with instructions to generate a simpler query. This works ~70% of the time. The remaining 30% results in a graceful error message to the user rather than a crash.

### The SQL retry pattern

```python
max_attempts = 3
error_history = []
current_sql = sql

for attempt in range(max_attempts):
    try:
        rows = execute_sql(current_sql)
        result_str = json.dumps(rows, default=str) if rows else "No results found."
        return {"sql_query": current_sql, "sql_result": result_str}
    except Exception as e:
        error_history.append(f"Attempt #{attempt + 1}: {e}")
        current_sql = retry_generate_sql(question, current_sql, error_history)
```

### What I'd change

The `run_sql_sub` function (used in the hybrid path) duplicates this entire pipeline inline — generate, retry loop, execute, summarize. If I were rebuilding, I'd extract a shared `_execute_sql_query(question, chat_history) -> dict` helper and use it from both paths.

### The table schema (passed to every SQL prompt)

```python
TABLE_SCHEMA = """
Table: customers (columns: customer_id TEXT, ...)
Table: orders (columns: order_id TEXT, customer_id TEXT, order_status TEXT, ...)
Table: order_items (columns: order_id TEXT, product_id TEXT, price TEXT, ...)
Table: products (columns: product_id TEXT, product_category_name TEXT, ...)
Table: product_category_name_translation (columns: product_category_name TEXT, product_category_name_english TEXT)
...
"""
```

---

## 10. Hybrid path — parallel RAG + SQL

The hybrid path handles compound questions that need both document retrieval and database queries. Example: "What is the refund policy and how many customers do we have?"

### The flow

```
decompose_question (LLM splits into sub-questions)
  ↓
route_sub_questions (LangGraph Send for parallel execution)
  ↓
run_rag_sub + run_sql_sub (executed in parallel)
  ↓
synthesise_hybrid (merge all answers into one response)
```

### Decomposition

The LLM splits the question into sub-questions, each tagged with a type and unique ID:

```json
{
  "sub_questions": [
    {"id": "doc_0", "question": "What is the refund policy?", "type": "document"},
    {"id": "sql_0", "question": "How many customers do we have?", "type": "database"}
  ]
}
```

### Parallel execution with `Send`

LangGraph's `Send` enables conditional fan-out:

```python
def route_sub_questions(state):
    sends = []
    for sq in state["sub_questions"]:
        if sq["type"] == "document":
            sends.append(Send("run_rag_sub", {**state, "sub_question": sq}))
        else:
            sends.append(Send("run_sql_sub", {**state, "sub_question": sq}))
    return sends
```

Each sub-question gets its own copy of the state (with reset loop counters), executed concurrently. The `Annotated[Sequence[tuple[str, str]], operator.add]` annotation on `sub_results` collects all parallel outputs.

### Fan-in

```python
def synthesise_hybrid(state):
    sub_results = state.get("sub_results", [])
    partial_answers = "\n\n".join(f"{k}: {v}" for k, v in sub_results)
    out = _llm.invoke(
        synthesise_hybrid_prompt.format(
            question=state["question"],
            partial_answers=partial_answers,
        )
    )
    return {"answer": out.content, "issup": "fully_supported", "isuse": "useful"}
```

### What I learned

- **Sub-questions need their own loop counters** — Without resetting `retries`, `rewrite_tries`, `docs`, etc., a RAG sub-question inherits state from a previous SQL sub-question.
- **`Send` expects a complete state dict** — You can't pass just the sub-question. You spread the current state and override the relevant fields.
- **The merge prompt is remarkably simple** — "Combine into a single natural response" is enough. The LLM handles the rest.

---

## 11. Conversation path — routing emotional messages

This was the last query type I added, and it was the simplest — but it had the biggest impact on user experience.

### The problem

Without a conversation path, emotional messages like "Thanks", "This is frustrating", or "That didn't help" went through the full RAG pipeline (retrieve → IsSUP → revise → IsUSE → rewrite) before failing gracefully. That's 10-15 LLM calls for a message that should just be acknowledged.

### The solution

A fourth query type that routes to a single `generate_direct` node — one LLM call, no retrieval, no validation loop.

```python
# In classifier prompt:
"- \"conversation\": User is expressing emotion, feedback, or conversational intent
  rather than requesting information. Examples: 'This is frustrating', 'Thanks',
  'Okay', 'That didn't help', 'I want a human'."

# In router:
elif qt == "conversation":
    return "generate_direct"
```

### The generate_direct node

```python
def generate_direct(state: State):
    chat_history = format_chat_history(state.get("chat_history", []))
    out = _llm.invoke(
        direct_generation_prompt.format(
            question=state["question"],
            chat_history=chat_history,
        )
    )
    return {
        "answer": out.content,
        "issup": "fully_supported",
        "isuse": "useful",
        "use_reason": "Directly answered from general knowledge or conversational response.",
    }
```

### Critical: quality signals must be set

Notice `issup="fully_supported"` and `isuse="useful"`. These prevent the escalation check from treating a simple "Thanks" acknowledgment as an unresolved issue. Without them, the escalation LLM would read the user's "This is frustrating" and the system's "I understand your concern" and escalate — even though the system responded appropriately.

### What I learned

- **The conversation path saves 10-15 LLM calls per emotional message** — That's 10-15x cost reduction for a common user behavior.
- **The quality signals are critical** — Without them, the escalation prompt has nothing to distinguish "user venting frustration and system acknowledged it" from "user asking a question and getting a bad answer."
- **This path didn't need a new pattern** — It uses the same `check_escalation` convergence as every other path.

---

## 12. Escalation — the hardest part to tune

Escalation detection sounds simple: check if the user is frustrated and route to a human. In practice, it was the most finicky part of the system.

### The escalation prompt

```python
escalation_check_prompt = ChatPromptTemplate.from_messages([
    ("system", """Determine whether to escalate to a human agent.

HARD TRIGGERS (immediate):
- User explicitly requests a human
- User files a complaint
- User expresses strong frustration

SOFT TRIGGERS (evaluate):
- Negative sentiment persists across 3+ turns
- Same issue remains unresolved after multiple attempts

Answer Quality Signals:
- issup: {issup}
- isuse: {isuse}

CRITICAL RULE: If issup is 'fully_supported' AND isuse is 'useful',
the question was successfully answered. Do NOT escalate for 'unresolved_issue'.
"""),
    ("human", "Conversation History:\n{chat_history}\n\nQuestion: {question}\nAnswer: {answer}\nissup: {issup}\nisuse: {isuse}"),
])
```

### The bug that took me the longest to find

Early versions of the escalation system escalated **every conversation with a frustrated user**, even when the system had answered their question correctly.

Example: User asks "How long does shipping take?" The system answers correctly. User says "That's too slow, this is frustrating." The escalation LLM sees "frustration" in the conversation and escalates — even though the system did its job.

**The fix**: Pass `issup` and `isuse` into the escalation prompt, and add the critical rule: "If the answer was fully supported and useful, do not escalate for unresolved_issue."

This required every answer-producing node to set these signals:

| Node | issup | isuse |
|------|-------|-------|
| RAG generate_from_context | Set by IsSUP node | Set by IsUSE node |
| SQL summarize_sql_result | `fully_supported` | `useful` |
| Hybrid synthesise_hybrid | `fully_supported` | `useful` |
| Conversation generate_direct | `fully_supported` | `useful` |

### The handoff summary

When escalation triggers, the system generates a structured handoff summary for the human agent:

```
User Goal
-----------
What the user was trying to do

System Actions
-----------
Recent documents consulted: [titles]
Recent SQL queries executed: [queries]

Current Answer
-----------
What the system responded

Escalation Reason
-----------
Why the system escalated
```

### What I learned

- **Escalation without quality signals is useless** — The LLM will escalate everything that looks even slightly negative.
- **The `reason` field is a `Literal`** — `human_requested`, `complaint`, `frustration`, `repeated_negative_sentiment`, `unresolved_issue`, `none`. This makes the reason actionable (e.g., different severity levels get different email styling).
- **The email is sent inside the graph** — The `generate_handoff` node calls `send_escalation_email()` and catches any SMTP errors. This keeps the email out of the critical path — if Gmail is down, the graph still completes.

---

## 13. Email — deterministic HTML alerts

The escalation email is built from structured data, not generated by an LLM. This was an important design decision.

### Why deterministic

An LLM-generated email might say "The user was frustrated about shipping" in HTML. A deterministic function builds the same HTML every time, with:

- Severity badge (color-coded by reason)
- Ticket ID (timestamp-based, generated once)
- Metadata table (reason, severity, message count, timestamp)
- Documents consulted (deduplicated, HTML-escaped)
- SQL queries executed (truncated to 300 chars, all last 5)
- Recommended action (mapped deterministically from reason)

### The key decisions

**One ticket ID** — Generated once for the email, used in both the subject line and the body. Prevents clock-tick mismatches.

**HTML escaping** — All user-provided content (summaries, doc titles, SQL queries) is passed through `html.escape()`. Prevents `<>` and `&` from breaking the email format.

**SQL truncation** — Each query is truncated to 300 characters. Generates clean, readable emails even for complex queries.

**No chat history in the email** — The summary captures the user's goal and the system's response. Full chat history would make the email too long and dilute the signal.

### The email signature

```python
def send_escalation_email(
    handoff_summary: str,
    reason: str,
    rag_docs_used: list[str] | None = None,
    sql_queries_executed: list[str] | None = None,
    message_count: int = 0,
) -> None:
```

All parameters are deterministic — no LLM calls, no ambiguity.

---

## 14. What I learned the hard way

### 1. State first, always

I started with nodes and added state fields as I went. This led to a bloated state (28 fields, 1 unused). **Design the state first. Every field should appear in at least one node's output and one node's input.**

### 2. One LLM call per node

Early versions had nodes that called the LLM multiple times (e.g., classify + route in the same function). This made debugging harder because I couldn't tell which call produced which output. **One node, one LLM call.**

### 3. Set quality signals on every answer

The escalation system depends on `issup` and `isuse` being set correctly. If a node produces an answer but doesn't set these, the escalation check will receive empty values and may escalate incorrectly. **Every answer-producing node sets issup="fully_supported" and isuse="useful" at minimum.**

### 4. Test each path independently

I tested the RAG path thoroughly before adding SQL, tested SQL before hybrid, and so on. When I finally added escalation, it worked on the first try on every path I'd already tested. **Integration bugs hide in the gaps between paths. Test each path alone before testing them together.**

### 5. The LLM will invent columns

SQL generation is the most hallucination-prone part of the system. The only fix is aggressive, repeated instructions in every prompt: "Verify every column name against the schema. If it doesn't appear verbatim, don't use it."

### 6. Don't invent filter values

The LLM loves creating realistic-looking IDs. The fix: "If the user didn't specify a concrete value, omit that filter rather than fabricating one."

### 7. Graphs are harder to debug than notebooks

In a notebook, you can inspect every variable after every cell. In a graph, state flows through nodes and you only see the final output. **Log everything.** Every node appends to `logs`. This is how I debug most issues.

### 8. The `Send` API requires the full state

When fanning out for hybrid questions, each branch gets its own copy of the state. If you forget to reset loop counters (`retries`, `rewrite_tries`, `docs`), one sub-question's state bleeds into another's.

---

## 15. What I'd keep the same

| Decision | Why I'd keep it |
|----------|----------------|
| **Single LangGraph, not separate agents** | Cleaner routing, shared state, one entry point. No need for four separate services. |
| **Pydantic models for structured output** | Catches parse errors early. Every LLM call returns a typed Python object, not raw JSON. |
| **Chat history as a list of dicts** | Simple, serializable, works with Streamlit session state. No need for a message broker or database. |
| **Compiled RAG subgraph** | Enables clean parallel execution for hybrid questions without duplicating 10 nodes. |
| **Convergence at escalation** | All four paths meet at one node. This is the key architectural insight — classify diverges, escalation converges. |
| **Deterministic email content** | No LLM in the email path. Structured data → HTML template → send. Reliable. |
| **Token streaming via callbacks** | The `TokenCollector` pattern is simple and works for both CLI and Streamlit without threads. |
| **Direct pymysql (no ORM)** | Simple, fast, no abstraction overhead. Table schema is passed to the LLM directly. |
| **FAISS for static PDFs** | Zero network latency, no API cost, takes 2 seconds to build. Pinecone would add complexity with no benefit for 3 PDFs. |

---

## 16. What I'd change

| Change | Why | Effort |
|--------|-----|--------|
| **Remove `decide_retrieval` from RAG subgraph** | The classifier already routes document questions. This node just re-checks. Saves 1 LLM call per query. | Low |
| **Extract shared SQL helper** | `run_sql_sub` duplicates the entire SQL pipeline inline. A `_execute_sql_query()` helper would eliminate ~60 lines of duplication. | Medium |
| **Merge `sql_retry_prompt` into `generate_sql_prompt`** | 90% identical content. Add "generate a simpler corrected query" as a conditional instruction instead. | Low |
| **Remove dead `conversation_summary` from state** | In my case it was defined but never read. Don't add fields until a node needs them. | Low |
| **Document state fields with a table** | Who sets what, who reads what, which path uses it. Would have caught unused fields early. | Low (documentation only) |
| **Pre-build and commit FAISS index** | Eliminates 60s cold start on `docker compose up`. Build once, commit as `faiss_index/`, load on startup. | Low |

---

## 17. Production considerations

### Health check endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vector_store": "ready",
        "database": "ready",
        "llm": "ready",
    }
```

Configure the ALB target group to use `/health` as the health check path. This ensures requests are only routed when the container is fully initialized.

### Rate limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    ...
```

10 requests per minute per IP prevents runaway billing from OpenRouter if the endpoint is accidentally bombarded.

### Secrets management

All sensitive values (OpenRouter API key, MySQL password, Gmail credentials) go in AWS Secrets Manager in production. The `.env` file is used for local development only and is gitignored.

```json
// ECS task definition secrets block
{
  "secrets": [
    { "name": "OPENROUTER_API_KEY", "valueFrom": "arn:aws:secretsmanager:..." },
    { "name": "MYSQL_PASSWORD", "valueFrom": "arn:aws:secretsmanager:..." }
  ]
}
```

### FAISS index loading

Build the FAISS index locally (one-time), commit it to the repo. On startup, load it from disk:

```python
vector_store = FAISS.load_local("app/faiss_index/", embeddings)
```

No 60s cold start during deployment or `docker compose up`.

### Dockerfile

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### ECS Fargate architecture

```
Internet → ALB (HTTPS) → ECS Fargate (1+ tasks) → FastAPI → LangGraph
                                                           → TiDB Cloud
                                                           → OpenRouter API
```

---

## 18. Checklist for building similar systems

### Phase 1 — Design

- [ ] Define state TypedDict with every field named and typed
- [ ] Create a table: which nodes set each field, which nodes read each field
- [ ] Define Pydantic models for every structured LLM output
- [ ] Write all prompts (at least draft versions)
- [ ] Draw the graph flow: what nodes connect to what, where routing happens

### Phase 2 — Implement

- [ ] Implement nodes as pure functions on state
- [ ] Log everything (every node appends to a `logs` list)
- [ ] Build and compile the graph
- [ ] Test each path independently (document → RAG, database → SQL, hybrid, conversation)
- [ ] Add escalation check last (it depends on quality signals from every other path)

### Phase 3 — Harden

- [ ] Set quality signals (`issup`, `isuse`) on every answer-producing node
- [ ] Add error handling for SQL execution (retry with error feedback)
- [ ] Add SQL query validation (column verification in prompts)
- [ ] Test escalation with known edge cases (frustration + good answer, complaint, thanks)

### Phase 4 — Ship

- [ ] Health check endpoint
- [ ] Rate limiting
- [ ] Dockerfile + docker-compose.yml
- [ ] Minimal input validation (max_length on question)
- [ ] Secrets management (no secrets in code or config)
- [ ] LangSmith tracing for debugging

---

## 19. The short version — TL;DR

**Planning > State > Models > Prompts > Nodes > Graph > Builder**

- State first. Every field must be read by one node and written by another.
- One LLM call per node. More nodes = easier debugging.
- Set quality signals on every answer. Escalation needs them.
- Test each path independently. Integration bugs hide in the gaps.
- Log everything. It's the only window into a running graph.
- Converge at escalation. Classify diverges, escalation converges.
- Deterministic email. No LLM in the critical path.
- FAISS not Pinecone. 3 PDFs don't need a vector database.
- The conversation path saves 10-15x LLM calls for emotional messages.
- The SQL path will be the buggiest. Invest in prompt engineering there.

---

*Built with LangGraph, OpenRouter GPT-4o-mini, FAISS, TiDB Cloud, PyMySQL, Streamlit, FastAPI, Docker, and Gmail SMTP.*
