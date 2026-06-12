import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.graph.builder import build_app
from app.state import State
from app.stream_events import TokenCollector

st.set_page_config(page_title="Self-RAG MCP", page_icon="🤖")
st.title("Self-RAG MCP — Company Document & Database Q&A")
st.caption("Ask about company policies, or query e-commerce data (customers, orders, products, etc.)")

with st.sidebar:
    if st.button("Talk to a Human"):
        st.session_state.force_escalation = True
        st.rerun()

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
            collector = TokenCollector()
            answer_text = ""
            answer_placeholder = st.empty()

            with progress_placeholder.container():
                log_area = st.empty()
                log_area.markdown("**Thinking process:**\n\n_starting..._")

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

                    log_text = "\n".join(f"- {l}" for l in collected_logs[-12:])
                    log_area.markdown(f"**Thinking process:**\n\n{log_text}")

                # Drain any tokens accumulated since last superstep
                while not collector._queue.empty():
                    token = collector._queue.get_nowait()
                    answer_text += token

                if answer_text:
                    answer_placeholder.markdown(answer_text + "▌")

            collector.mark_done()

            # Drain remaining tokens
            while not collector._queue.empty():
                token = collector._queue.get_nowait()
                answer_text += token

            result = final_state or {}
            query_type = result.get("query_type", "document")

            if query_type in ("database", "hybrid"):
                final_answer = result.get("db_answer") or result.get("answer", "No answer found.")
            else:
                final_answer = result.get("answer", "No answer found.")

            # Show final answer (prefer accumulated tokens, fall back to state answer)
            final_display = answer_text or final_answer
            answer_placeholder.markdown(final_display)

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

            # Escalation banner (normal escalation during graph run)
            if result.get("escalated"):
                st.warning(f"🤝 Escalated to human support — reason: **{result['escalation_reason']}**")
                summary = result.get("handoff_summary", "")
                if summary:
                    with st.expander("Handoff summary for agent"):
                        st.markdown(summary)

            debug = {}

            if query_type == "hybrid":
                debug = {
                    "query_type": "hybrid",
                    "sub_questions": result.get("sub_questions", []),
                    "sub_results": dict(result.get("sub_results", [])),
                    "logs": collected_logs,
                }
            elif query_type == "database":
                debug = {
                    "query_type": "database",
                    "sql_query": result.get("sql_query", ""),
                    "sql_result_preview": (result.get("sql_result", "")[:500] + "...") if len(result.get("sql_result", "")) > 500 else result.get("sql_result", ""),
                    "db_error": result.get("db_error", ""),
                    "logs": collected_logs,
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
                    "logs": collected_logs,
                }

            with st.expander("Debug info"):
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

        st.session_state.history.append({"role": "assistant", "content": final_answer, "debug": debug if 'debug' in locals() else {}})
