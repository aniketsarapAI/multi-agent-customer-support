import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.graph.builder import build_app
from app.state import State
from app.stream_events import TokenCollector




def _format_step(log: str) -> str | None:
    m = re.search(r"✅ classify_question:\s*(\S+)", log)
    if m:
        return f"Classified as **{m.group(1).upper()}**"

    m = re.search(r"🔍 retrieve: found (\d+) document", log)
    if m:
        return f"Retrieved **{m.group(1)}** documents"

    m = re.search(r"✅ is_relevant:\s*(\d+)/(\d+) relevant", log)
    if m:
        return f"**{m.group(1)} of {m.group(2)}** documents relevant"

    if "✅ generate_from_context: done" in log:
        return "Answer drafted from evidence"

    m = re.search(r"✅ is_sup:\s*(\S+)", log)
    if m:
        return f"Support verification: **{m.group(1).upper()}**"

    m = re.search(r"✅ revise_answer: done \(attempt (\d+)\)", log)
    if m:
        return f"Answer revised (attempt {m.group(1)})"

    m = re.search(r"✅ is_use:\s*(\S+)\s*[—–-]\s*(.+)", log)
    if m:
        return f"Usefulness: **{m.group(1).upper()}** — {m.group(2)}"

    m = re.search(r"✅ rewrite_sql_query:\s*→\s*(.+)", log)
    if m:
        return f"Refined SQL query → {m.group(1)[:80]}"

    if "✅ generate_sql: SQL generated" in log:
        return "SQL query generated"

    m = re.search(r"✅ execute_sql_node:\s*(\d+) row", log)
    if m:
        return f"Query returned **{m.group(1)}** rows"
    if "✅ execute_sql_node: no results" in log:
        return "Query returned no results"

    if "✅ summarize_sql_result: done" in log:
        return "SQL results summarized"

    if "✅ synthesise_hybrid: done" in log:
        return "Hybrid answers merged"

    m = re.search(r"✅ decompose_question:\s*→\s*(\d+) sub-question", log)
    if m:
        return f"Question decomposed into **{m.group(1)}** sub-questions"

    if "✅ generate_direct: done" in log:
        return "Answered directly"

    m = re.search(r"✅ run_rag_sub\((\w+)\): done", log)
    if m:
        return f"RAG sub-query **{m.group(1)}** complete"

    m = re.search(r"✅ run_sql_sub\((\w+)\): done", log)
    if m:
        return f"SQL sub-query **{m.group(1)}** complete"

    if "⚠️ no_answer_found" in log:
        return "No relevant documents found"

    if "✅ visualize_sql_result: chart generated" in log:
        return "Chart generated from results"

    m = re.search(r"❌ execute_sql_node: all \d+ attempts failed", log)
    if m:
        return "SQL execution failed after retries"

    if "✅ generate_handoff: done" in log:
        return "Handoff summary generated"

    m = re.search(r"\s*escalate=(\S+)\s+reason=(\S+)", log)
    if m:
        if m.group(1) == "False":
            return "Escalation check: not required"
        return f"Escalation required: **{m.group(2).upper()}**"

    # Fallback: show raw completion logs
    if log.startswith("✅") or log.startswith("⚠️") or log.startswith("❌"):
        cleaned = re.sub(r"^[✅⚠️❌]\s*", "", log).strip()
        return cleaned[:100].capitalize()

    return None


def _format_timeline(logs: list[str]) -> str:
    step = 0
    lines = []
    for log in logs:
        formatted = _format_step(log)
        if formatted is None:
            continue
        step += 1
        lines.append(f"**{step}.** {formatted}")
    if not lines:
        return "_waiting..._"
    return "\n\n".join(lines)


st.set_page_config(page_title="Self-RAG MCP", page_icon="🤖")
st.title("Self-RAG MCP — Company Document & Database Q&A")
st.caption("Ask about company policies, or query e-commerce data (customers, orders, products, etc.)")

with st.sidebar:
    if st.button("Talk to a Human"):
        st.session_state.force_escalation = True
        st.rerun()
    st.divider()
    st.markdown("**View Mode**")
    # Demo/developer visibility controls.
    # Production deployments should default both to False.
    st.checkbox("Show agent execution trace", value=True, key="show_trace")
    st.checkbox("Show debug panel", value=False, key="show_debug")
    if st.session_state.show_trace or st.session_state.show_debug:
        st.caption("Demo / developer features enabled")

if "app" not in st.session_state:
    with st.spinner("Loading documents and building graph..."):
        st.session_state.app = build_app()
        st.session_state.app_ready = True

if "history" not in st.session_state:
    st.session_state.history = []

if "rag_docs_used" not in st.session_state:
    st.session_state.rag_docs_used = []
if "sql_queries_executed" not in st.session_state:
    st.session_state.sql_queries_executed = []
if "show_trace" not in st.session_state:
    st.session_state.show_trace = True
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "debug" in msg:
            with st.expander("Debug info"):
                st.json(msg["debug"])

# ── Immediate handoff on button click ──
force_esc = st.session_state.pop("force_escalation", False)
if force_esc and st.session_state.history:
    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
    with st.spinner("Generating handoff summary..."):
        handoff_state: State = {
            "question": "I need to speak to a human agent right now",
            "original_question": chat_history[-1]["content"],
            "query_type": "conversation",
            "chat_history": chat_history,
            "sub_questions": [], "sub_results": [], "sub_question": None,
            "retrieval_query": "", "rewrite_tries": 0, "need_retrieval": False,
            "docs": [], "relevant_docs": [], "context": "", "answer": "",
            "issup": "fully_supported", "evidence": [], "retries": 0,
            "isuse": "useful", "use_reason": "",
            "logs": [], "sql_query": "", "sql_result": "",
            "db_answer": "", "db_error": "", "visualization_spec": None,
            "escalated": False, "escalation_reason": "", "handoff_summary": "",
            "rag_docs_used": st.session_state.rag_docs_used,
            "sql_queries_executed": st.session_state.sql_queries_executed,
        }
        result = None
        for output in st.session_state.app.stream(handoff_state, stream_mode="values"):
            result = output
        summary = (result or {}).get("handoff_summary", "User requested handoff.")
    with st.chat_message("assistant"):
        st.warning("🤝 Escalated to human support — reason: **human_requested**")
        with st.expander("Handoff summary for agent"):
            st.markdown(summary)

if prompt := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        progress_placeholder = st.empty()
        placeholder.markdown("Thinking...")

        chat_history = []
        for msg in st.session_state.history:
            chat_history.append({"role": msg["role"], "content": msg["content"]})

        initial_state: State = {
            "question": prompt,
            "original_question": prompt,
            "query_type": "document",
            "chat_history": chat_history,
            "sub_questions": [],
            "sub_results": [],
            "sub_question": None,
            "retrieval_query": "",
            "rewrite_tries": 0,
            "need_retrieval": False,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "issup": "no_support",
            "evidence": [],
            "retries": 0,
            "isuse": "not_useful",
            "use_reason": "",
            "logs": [],
            "sql_query": "",
            "sql_result": "",
            "db_answer": "",
            "db_error": "",
            "visualization_spec": None,
            "escalated": False,
            "escalation_reason": "",
            "handoff_summary": "",
            "rag_docs_used": st.session_state.rag_docs_used,
            "sql_queries_executed": st.session_state.sql_queries_executed,
        }

        try:
            collected_logs: list[str] = []
            final_state = None

            # TokenCollector preserved for future FastAPI/SSE use — not rendered
            collector = TokenCollector()
            debug = {}

            with progress_placeholder.container():
                trace_area = st.empty()
                if st.session_state.show_trace:
                    trace_area.markdown("**🧠 Agent Execution**\n\n_starting..._")

            for output in st.session_state.app.stream(
                initial_state,
                stream_mode="values",
                config={"callbacks": [collector]},
            ):
                final_state = output
                new_logs = output.get("logs", [])
                if isinstance(new_logs, list):
                    for log in new_logs:
                        if log not in collected_logs:
                            collected_logs.append(log)

                    if st.session_state.show_trace:
                        timeline = _format_timeline(collected_logs)
                        trace_area.markdown(f"**🧠 Agent Execution**\n\n{timeline}")

                # Drain tokens (preserved for future SSE) — not used for display
                while not collector._queue.empty():
                    collector._queue.get_nowait()

            collector.mark_done()
            while not collector._queue.empty():
                collector._queue.get_nowait()

            result = final_state or {}
            query_type = result.get("query_type", "document")

            if query_type in ("database", "hybrid"):
                final_answer = result.get("db_answer") or result.get("answer", "No answer found.")
            else:
                final_answer = result.get("answer", "No answer found.")

            # ── Section 1: Final Answer ──
            placeholder.markdown(final_answer)

            # ── Section 2: Agent Execution timeline (collapsed after completion) ──
            if st.session_state.show_trace and collected_logs:
                timeline = _format_timeline(collected_logs)
                with st.expander("🧠 Agent Execution", expanded=False):
                    st.markdown(timeline)

            # ── Visualization ──
            viz_spec = result.get("visualization_spec")
            sql_raw = result.get("sql_result", "")
            if viz_spec and sql_raw:
                try:
                    import json
                    chart_data = json.loads(sql_raw)
                    if chart_data and isinstance(chart_data, list):
                        st.vega_lite_chart(chart_data, viz_spec, use_container_width=True)
                except Exception:
                    pass

            # ── Escalation banner ──
            if result.get("escalated"):
                st.warning(f"🤝 Escalated to human support — reason: **{result['escalation_reason']}**")
                summary = result.get("handoff_summary", "")
                if summary:
                    with st.expander("Handoff summary for agent"):
                        st.markdown(summary)

            # ── Section 3: Debug Panel (always collapsed) ──
            if st.session_state.show_debug:
                if query_type == "hybrid":
                    debug = {
                        "query_type": "hybrid",
                        "sub_questions": result.get("sub_questions", []),
                        "sub_results": dict(result.get("sub_results", [])),
                    }
                elif query_type == "database":
                    debug = {
                        "query_type": "database",
                        "sql_query": result.get("sql_query", ""),
                        "sql_result_preview": (
                            (result.get("sql_result", "")[:500] + "...")
                            if len(result.get("sql_result", "")) > 500
                            else result.get("sql_result", "")
                        ),
                        "db_error": result.get("db_error", ""),
                    }
                else:
                    debug = {
                        "query_type": result.get("query_type", "document"),
                        "need_retrieval": result.get("need_retrieval"),
                        "rewrite_tries": result.get("rewrite_tries", 0),
                        "retries": result.get("retries", 0),
                        "retrieved_docs": len(result.get("docs", []) or []),
                        "relevant_docs": len(result.get("relevant_docs", []) or []),
                        "issup": result.get("issup"),
                        "evidence": result.get("evidence", []),
                        "isuse": result.get("isuse"),
                        "use_reason": result.get("use_reason", ""),
                    }

                debug["escalated"] = result.get("escalated", False)
                debug["escalation_reason"] = result.get("escalation_reason", "")

                with st.expander("Debug Info"):
                    st.json(debug)

            # Accumulate retrieval/sql metadata across turns
            new_docs = [
                d.metadata.get("title") or d.metadata.get("source", "")
                for d in result.get("relevant_docs", []) or []
                if d is not None
            ]
            new_docs = [t for t in new_docs if t]
            sql_raw = result.get("sql_query", "")
            new_sql = [sql_raw.replace("\n", " ")[:150]] if sql_raw else []
            docs = list(dict.fromkeys(st.session_state.rag_docs_used + new_docs))[-5:]
            queries = (st.session_state.sql_queries_executed + new_sql)[-5:]
            st.session_state.rag_docs_used = docs
            st.session_state.sql_queries_executed = queries
        except Exception as e:
            placeholder.error(f"Error: {e}")
            final_answer = f"Error: {e}"

        st.session_state.history.append({
            "role": "assistant",
            "content": final_answer,
            "debug": debug if st.session_state.show_debug else {},
        })
