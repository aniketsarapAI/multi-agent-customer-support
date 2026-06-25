import json

from langgraph.graph import StateGraph, START, END

from app.models.state import SQLAgentState
from app.models.planning import SQLQueryDecision, SQLRewriteDecision
from app.prompts import (
    sql_rewrite_prompt,
    generate_sql_prompt,
    sql_retry_prompt,
    summarize_sql_prompt,
)
from app.infrastructure.db_agent import execute_sql, TABLE_SCHEMA
from app.chat_history import format_chat_history


def make_rewrite_sql_query(llm):
    def rewrite_sql_query(state: SQLAgentState):
        logs = ["🔍 rewrite_sql_query: refining vague query..."]
        chat_history = format_chat_history(state.get("chat_history", []))
        structured = llm.with_structured_output(SQLRewriteDecision)
        decision: SQLRewriteDecision = structured.invoke(
            sql_rewrite_prompt.format_messages(
                question=state["question"],
                chat_history=chat_history,
            )
        )
        logs.append(f"✅ rewrite_sql_query: → {decision.refined_query[:80]}...")
        return {"question": decision.refined_query, "logs": logs}
    return rewrite_sql_query


def make_generate_sql(llm):
    def generate_sql(state: SQLAgentState, error: str = "", bad_sql: str = ""):
        structured = llm.with_structured_output(SQLQueryDecision)
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
    return generate_sql


def make_execute_sql_node(llm):
    _generate_sql = make_generate_sql(llm)

    def execute_sql_node(state: SQLAgentState):
        logs = ["🔍 execute_sql_node: running query..."]
        sql = state.get("sql_query", "")
        if not sql:
            logs.append("❌ execute_sql_node: no SQL query")
            return {"db_error": "No SQL query generated.", "answer": "Failed to generate a valid SQL query.", "logs": logs}

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
                    retry = _generate_sql(state, error=context, bad_sql=current_sql)
                    new_sql = retry.get("sql_query", "")
                    if not new_sql or new_sql == current_sql:
                        logs.append(f"❌ execute_sql_node: retry #{attempt + 1} produced no improvement")
                    current_sql = new_sql or current_sql
                else:
                    logs.append(f"❌ execute_sql_node: all {max_attempts} attempts failed — {error_msg}")
                    return {"db_error": error_msg, "answer": f"Error executing query: {error_msg}", "logs": logs}

        return {"db_error": error_history[-1] if error_history else "Unknown error", "answer": "Failed to execute query after retries.", "logs": logs}
    return execute_sql_node


def visualize_sql_result(state: SQLAgentState):
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

    if not isinstance(rows[0], dict):
        logs.append("⚠️ visualize_sql_result: result is not dict rows")
        return {"visualization_spec": None, "logs": logs}

    numeric_cols = []
    cat_cols = []
    for col in rows[0].keys():
        vals = [r.get(col) for r in rows if r.get(col) is not None]
        if not vals:
            continue
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

    spec = {
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": cat_cols[0], "type": "nominal", "sort": "-y", "title": cat_cols[0]},
            "y": {"field": numeric_cols[0], "type": "quantitative", "title": numeric_cols[0]},
        },
    }

    logs.append(f"✅ visualize_sql_result: chart generated ({cat_cols[0]} vs {numeric_cols[0]})")
    return {"visualization_spec": spec, "logs": logs}


def make_summarize_sql_result(llm):
    def summarize_sql_result(state: SQLAgentState):
        logs = ["🔍 summarize_sql_result: creating answer from SQL results..."]
        if state.get("db_error"):
            logs.append("⚠️ summarize_sql_result: skipping due to prior error")
            return {"answer": state.get("answer", "An error occurred."), "logs": logs}

        out = llm.invoke(
            summarize_sql_prompt.format_messages(
                question=state["question"],
                sql_query=state.get("sql_query", ""),
                sql_result=state.get("sql_result", "No results."),
            )
        )
        logs.append("✅ summarize_sql_result: done")
        return {
            "answer": out.content,
            "logs": logs,
            "issup": "fully_supported",
            "isuse": "useful",
            "use_reason": "Answer generated from SQL query results.",
        }
    return summarize_sql_result


def build_sql_subgraph(llm):
    g = StateGraph(SQLAgentState)

    g.add_node("rewrite_sql_query", make_rewrite_sql_query(llm))
    g.add_node("generate_sql", make_generate_sql(llm))
    g.add_node("execute_sql", make_execute_sql_node(llm))
    g.add_node("visualize_sql", visualize_sql_result)
    g.add_node("summarize_sql", make_summarize_sql_result(llm))

    g.add_edge(START, "rewrite_sql_query")
    g.add_edge("rewrite_sql_query", "generate_sql")
    g.add_edge("generate_sql", "execute_sql")
    g.add_edge("execute_sql", "visualize_sql")
    g.add_edge("visualize_sql", "summarize_sql")
    g.add_edge("summarize_sql", END)

    subgraph = g.compile()
    return subgraph
