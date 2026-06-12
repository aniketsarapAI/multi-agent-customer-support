import json
import ast
from typing import Literal

from langgraph.types import Send

from app.state import State
from app.models import (
    QueryTypeDecision,
    DecomposeDecision,
    SQLQueryDecision,
    SQLRewriteDecision,
)
from app.prompts import (
    classify_question_prompt_v2,
    decompose_question_prompt,
    sql_rewrite_prompt,
    generate_sql_prompt,
    sql_retry_prompt,
    summarize_sql_prompt,
    synthesise_hybrid_prompt,
)
from app.db_agent import execute_sql, TABLE_SCHEMA
from app.graph.rag_subgraph import build_rag_subgraph
from app.chat_history import format_chat_history

_llm = None
_rag_subgraph = None


def set_llm(llm):
    global _llm
    _llm = llm


def get_rag_subgraph():
    global _rag_subgraph
    if _rag_subgraph is None:
        _rag_subgraph = build_rag_subgraph()
    return _rag_subgraph


# ---------------------------------------------------------------------------
# Classify question
# ---------------------------------------------------------------------------
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


def route_after_classify(state: State) -> Literal["retrieve", "rewrite_sql_query", "decompose_question", "generate_direct"]:
    qt = state.get("query_type", "document")
    if qt == "database":
        return "rewrite_sql_query"
    elif qt == "hybrid":
        return "decompose_question"
    elif qt == "conversation":
        return "generate_direct"
    else:
        return "retrieve"


# ---------------------------------------------------------------------------
# Rewrite vague SQL questions
# ---------------------------------------------------------------------------
def rewrite_sql_query(state: State):
    logs = ["🔍 rewrite_sql_query: refining vague query..."]
    chat_history = format_chat_history(state.get("chat_history", []))
    structured = _llm.with_structured_output(SQLRewriteDecision)
    decision: SQLRewriteDecision = structured.invoke(
        sql_rewrite_prompt.format_messages(
            question=state["question"],
            chat_history=chat_history,
        )
    )
    logs.append(f"✅ rewrite_sql_query: → {decision.refined_query[:80]}...")
    return {"question": decision.refined_query, "logs": logs}


# ---------------------------------------------------------------------------
# Decompose compound question into sub-questions
# ---------------------------------------------------------------------------
def decompose_question(state: State):
    logs = ["🔍 decompose_question: splitting compound question..."]
    chat_history = format_chat_history(state.get("chat_history", []))
    structured = _llm.with_structured_output(DecomposeDecision)
    decision: DecomposeDecision = structured.invoke(
        decompose_question_prompt.format_messages(
            question=state["question"],
            chat_history=chat_history,
        )
    )
    n = len(decision.sub_questions)
    logs.append(f"✅ decompose_question: → {n} sub-question(s)")
    return {"sub_questions": decision.sub_questions, "logs": logs}


def route_sub_questions(state: State) -> list[Send]:
    sqs = state.get("sub_questions", [])
    if not sqs:
        return [Send("run_rag_sub", {**state, "sub_question": None})]

    sends = []
    for sq in sqs:
        sq_id = sq["id"] if isinstance(sq, dict) else sq.id
        sq_question = sq["question"] if isinstance(sq, dict) else sq.question
        sq_type = sq["type"] if isinstance(sq, dict) else sq.type
        sq_dict = {"id": sq_id, "question": sq_question, "type": sq_type}
        target = "run_rag_sub" if sq_type == "document" else "run_sql_sub"
        sends.append(
            Send(target, {
                **state,
                "sub_question": sq_dict,
                # Reset loop counters for each sub-question
                "answer": "",
                "db_answer": "",
                "sql_query": "",
                "sql_result": "",
                "db_error": "",
                "retrieval_query": "",
                "rewrite_tries": 0,
                "retries": 0,
                "docs": [],
                "relevant_docs": [],
                "context": "",
                "issup": "no_support",
                "evidence": [],
                "isuse": "not_useful",
                "use_reason": "",
                "logs": [],
            })
        )
    return sends


# ---------------------------------------------------------------------------
# Run RAG subgraph for a document sub-question
# ---------------------------------------------------------------------------
def run_rag_sub(state: State) -> State:
    sq = state["sub_question"]
    sub_id = sq["id"]
    logs = [f"🔍 run_rag_sub({sub_id}): searching documents..."]
    log_lines = logs

    subgraph = get_rag_subgraph()
    result_state = subgraph.invoke({
        **state,
        "question": sq["question"],
        "logs": [],
    })

    # Note: issup, evidence, isuse, use_reason from the subgraph
    # will overwrite parent state on exit. synthesise_hybrid does
    # not read them, so this is safe. Only sub_results is consumed downstream.
    sub_logs = result_state.get("logs", [])
    answer = result_state.get("answer") or "No answer found."
    log_lines.append(f"✅ run_rag_sub({sub_id}): done ({len(sub_logs)} steps)")
    return {"sub_results": [(sub_id, answer)], "logs": log_lines}


# ---------------------------------------------------------------------------
# Run SQL pipeline for a database sub-question
# ---------------------------------------------------------------------------
def run_sql_sub(state: State) -> State:
    sq = state["sub_question"]
    sub_id = sq["id"]
    logs = [f"🔍 run_sql_sub({sub_id}): generating SQL..."]

    chat_history = format_chat_history(state.get("chat_history", []))
    structured = _llm.with_structured_output(SQLQueryDecision)
    decision: SQLQueryDecision = structured.invoke(
        generate_sql_prompt.format_messages(
            question=sq["question"],
            table_schema=TABLE_SCHEMA,
            chat_history=chat_history,
        )
    )
    sql = decision.sql_query
    logs.append(f"🔍 run_sql_sub({sub_id}): executing query...")

    max_attempts = 3
    error_history = []
    current_sql = sql
    for attempt in range(max_attempts):
        try:
            rows = execute_sql(current_sql)
            result_str = json.dumps(rows, default=str) if rows else "No results found."
            sql = current_sql
            break
        except Exception as e:
            error_msg = str(e)
            error_history.append(f"Attempt #{attempt + 1}: {error_msg}")
            if attempt < max_attempts - 1:
                logs.append(f"⚠️ run_sql_sub({sub_id}): attempt #{attempt + 1} failed — {error_msg}")
                context = "\n".join(error_history)
                retry_decision: SQLQueryDecision = structured.invoke(
                    sql_retry_prompt.format_messages(
                        question=sq["question"],
                        table_schema=TABLE_SCHEMA,
                        bad_sql=current_sql,
                        error=context,
                    )
                )
                current_sql = retry_decision.sql_query
            else:
                logs.append(f"❌ run_sql_sub({sub_id}): all {max_attempts} attempts failed — {error_msg}")
                return {"sub_results": [(sub_id, f"Error: {error_msg}")], "logs": logs}
    else:
        logs.append(f"❌ run_sql_sub({sub_id}): all {max_attempts} attempts failed")
        return {"sub_results": [(sub_id, "Error: query execution failed after retries.")], "logs": logs}

    logs.append(f"🔍 run_sql_sub({sub_id}): summarizing result...")
    out = _llm.invoke(
        summarize_sql_prompt.format_messages(
            question=sq["question"],
            sql_query=sql,
            sql_result=result_str,
        )
    )
    logs.append(f"✅ run_sql_sub({sub_id}): done")
    return {"sub_results": [(sub_id, out.content)], "logs": logs}


# ---------------------------------------------------------------------------
# Synthesise hybrid results
# ---------------------------------------------------------------------------
def synthesise_hybrid(state: State):
    logs = ["🔍 synthesise_hybrid: merging partial answers..."]
    sub_results = state.get("sub_results", [])
    if not sub_results:
        logs.append("⚠️ synthesise_hybrid: no sub-results to merge")
        return {"answer": state.get("answer", "No answer found."), "logs": logs}

    partial_answers = "\n\n".join(
        f"{k}: {v}" for k, v in sub_results
    )

    out = _llm.invoke(
        synthesise_hybrid_prompt.format_messages(
            question=state["question"],
            partial_answers=partial_answers,
        )
    )
    logs.append("✅ synthesise_hybrid: done")
    return {
        "answer": out.content,
        "db_answer": out.content,
        "logs": logs,
        "issup": "fully_supported",
        "isuse": "useful",
        "use_reason": "Hybrid answer synthesized from sub-results.",
    }


# ---------------------------------------------------------------------------
# Original SQL agent nodes (single-question path, unchanged)
# ---------------------------------------------------------------------------
def generate_sql(state: State, error: str = "", bad_sql: str = ""):
    structured = _llm.with_structured_output(SQLQueryDecision)

    if error and bad_sql:
        decision: SQLQueryDecision = structured.invoke(
            sql_retry_prompt.format_messages(
                question=state["question"],
                table_schema=TABLE_SCHEMA,
                bad_sql=bad_sql,
                error=error,
            )
        )
    else:
        chat_history = format_chat_history(state.get("chat_history", []))
        decision: SQLQueryDecision = structured.invoke(
            generate_sql_prompt.format_messages(
                question=state["question"],
                table_schema=TABLE_SCHEMA,
                chat_history=chat_history,
            )
        )
    return {"sql_query": decision.sql_query, "logs": ["🔍 generate_sql: SQL generated"]}


def execute_sql_node(state: State):
    logs = ["🔍 execute_sql_node: running query..."]
    sql = state.get("sql_query", "")
    if not sql:
        logs.append("❌ execute_sql_node: no SQL query")
        return {"db_error": "No SQL query generated.", "db_answer": "Failed to generate a valid SQL query.", "logs": logs}

    max_attempts = 3
    error_history = []
    current_sql = sql

    for attempt in range(max_attempts):
        try:
            rows = execute_sql(current_sql)
            result_str = json.dumps(rows, default=str) if rows else "No results found."
            msg = f"✅ execute_sql_node: {len(rows)} row(s) returned" if rows else "✅ execute_sql_node: no results"
            logs.append(msg)
            if attempt > 0:
                logs.append(f"✅ execute_sql_node: succeeded on retry #{attempt}")
            return {"sql_query": current_sql, "sql_result": result_str, "logs": logs}
        except Exception as e:
            error_msg = str(e)
            error_history.append(f"Attempt #{attempt + 1}: {error_msg}")
            if attempt < max_attempts - 1:
                logs.append(f"⚠️ execute_sql_node: attempt #{attempt + 1} failed — {error_msg}")
                context = "\n".join(error_history)
                retry = generate_sql(state, error=context, bad_sql=current_sql)
                new_sql = retry.get("sql_query", "")
                if not new_sql or new_sql == current_sql:
                    logs.append(f"❌ execute_sql_node: retry #{attempt + 1} produced no improvement")
                current_sql = new_sql or current_sql
            else:
                logs.append(f"❌ execute_sql_node: all {max_attempts} attempts failed — {error_msg}")
                return {"db_error": error_msg, "db_answer": f"Error executing query: {error_msg}", "logs": logs}

    return {"db_error": error_history[-1] if error_history else "Unknown error", "db_answer": "Failed to execute query after retries.", "logs": logs}


# ---------------------------------------------------------------------------
# Visualization for SQL results
# ---------------------------------------------------------------------------
def visualize_sql_result(state: State):
    logs = ["🔍 visualize_sql_result: checking if chart is suitable..."]
    result_str = state.get("sql_result", "")
    if not result_str or result_str == "No results found.":
        logs.append("⚠️ visualize_sql_result: no data to visualize")
        return {"visualization_spec": None, "logs": logs}

    try:
        rows = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        logs.append("⚠️ visualize_sql_result: could not parse result as JSON")
        return {"visualization_spec": None, "logs": logs}

    if not isinstance(rows, list) or len(rows) == 0:
        logs.append("⚠️ visualize_sql_result: empty result set")
        return {"visualization_spec": None, "logs": logs}

    # Detect numeric and categorical columns
    if not isinstance(rows[0], dict):
        logs.append("⚠️ visualize_sql_result: result is not dict rows")
        return {"visualization_spec": None, "logs": logs}

    numeric_cols = []
    cat_cols = []
    for col in rows[0].keys():
        vals = [r.get(col) for r in rows if r.get(col) is not None]
        if not vals:
            continue
        # Check if column is numeric
        numeric_count = 0
        for v in vals:
            try:
                float(v)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        if numeric_count == len(vals):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    if not numeric_cols or not cat_cols:
        logs.append("⚠️ visualize_sql_result: no suitable chart columns found")
        return {"visualization_spec": None, "logs": logs}

    # Prefer bar chart for categorical x numeric
    spec = {
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": cat_cols[0], "type": "nominal", "sort": "-y", "title": cat_cols[0]},
            "y": {"field": numeric_cols[0], "type": "quantitative", "title": numeric_cols[0]},
        },
    }

    logs.append(f"✅ visualize_sql_result: chart generated ({cat_cols[0]} vs {numeric_cols[0]})")
    return {"visualization_spec": spec, "logs": logs}


def summarize_sql_result(state: State):
    logs = ["🔍 summarize_sql_result: creating answer from SQL results..."]
    if state.get("db_error"):
        logs.append("⚠️ summarize_sql_result: skipping due to prior error")
        return {"db_answer": state.get("db_answer", "An error occurred."), "logs": logs}

    out = _llm.invoke(
        summarize_sql_prompt.format_messages(
            question=state["question"],
            sql_query=state.get("sql_query", ""),
            sql_result=state.get("sql_result", "No results."),
        )
    )
    logs.append("✅ summarize_sql_result: done")
    return {
        "db_answer": out.content,
        "answer": out.content,
        "logs": logs,
        "issup": "fully_supported",
        "isuse": "useful",
        "use_reason": "Answer generated from SQL query results.",
    }
