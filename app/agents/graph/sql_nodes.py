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
from app.infrastructure.db_agent import (
    execute_sql,
    TABLE_SCHEMA,
    SQLValidationError,
    SQLSyntaxError,
    SQLConnectionError,
)
from app.chat_history import format_chat_history


def make_rewrite_sql_query(llm):
    def rewrite_sql_query(state: SQLAgentState):
        logs = ["🔍 rewrite_sql_query: refining vague query..."]
        chat_history = format_chat_history(state.get("chat_history", []))
        structured = llm.with_structured_output(
            SQLRewriteDecision, default=lambda: SQLRewriteDecision(refined_query="")
        )
        decision: SQLRewriteDecision = structured.invoke(
            sql_rewrite_prompt.format_messages(
                question=state["question"],
                chat_history=chat_history,
            )
        )
        refined = decision.refined_query or state["question"]
        logs.append(f"✅ rewrite_sql_query: → {refined[:80]}...")
        return {"question": refined, "logs": logs}
    return rewrite_sql_query


def make_generate_sql(llm):
    def generate_sql(state: SQLAgentState):
        structured = llm.with_structured_output(
            SQLQueryDecision, default=lambda: SQLQueryDecision(sql_query="")
        )
        logs = ["🔍 generate_sql: SQL generated"]
        error = state.get("db_error", "")
        bad_sql = state.get("sql_query", "")
        if error and bad_sql:
            logs = [f"🔍 generate_sql: retrying after error — {error[:80]}"]
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
        return {"sql_query": decision.sql_query, "logs": logs}
    return generate_sql


def make_execute_sql_node(llm):
    def execute_sql_node(state: SQLAgentState):
        logs = ["🔍 execute_sql_node: running query..."]
        sql = state.get("sql_query", "")
        if not sql:
            logs.append("❌ execute_sql_node: no SQL query")
            return {
                "db_error": "No SQL query generated.",
                "answer": "Failed to generate a valid SQL query.",
                "issup": "no_support",
                "isuse": "not_useful",
                "use_reason": "SQL generation failed.",
                "logs": logs,
            }

        try:
            rows = execute_sql(sql)
            result_str = json.dumps(rows, default=str) if rows else "No results found."
            logs.append(f"✅ execute_sql_node: {len(rows)} row(s) returned" if rows else "✅ execute_sql_node: no results")
            return {"sql_query": sql, "sql_result": result_str, "db_error": "", "logs": logs}
        except SQLValidationError as e:
            return {"db_error": str(e), "logs": logs + [f"❌ Validation failed: {e}"]}
        except SQLConnectionError as e:
            return {"db_error": str(e), "logs": logs + [f"❌ Connection error: {e}"]}
        except SQLSyntaxError as e:
            logs.append(f"⚠️ execute_sql_node: syntax error — {e}")
            return {"db_error": str(e), "logs": logs, "retry_count": state.get("retry_count", 0) + 1}
        except Exception as e:
            return {"db_error": str(e), "logs": logs + [f"❌ Unexpected error: {e}"]}
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
                question=state.get("original_question") or state.get("question", ""),
                sql_query=state.get("sql_query", ""),
                sql_result=state.get("sql_result", "No results."),
            )
        )
        logs.append("✅ summarize_sql_result: done")
        return {
            "answer": out.content,
            "logs": logs,
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

    def route_after_execute(state: SQLAgentState):
        if state.get("db_error") and state.get("retry_count", 0) < 3:
            return "generate_sql"
        return "visualize_sql"

    g.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {"generate_sql": "generate_sql", "visualize_sql": "visualize_sql"},
    )
    g.add_edge("visualize_sql", "summarize_sql")
    g.add_edge("summarize_sql", END)

    subgraph = g.compile()
    return subgraph
