import os
import re
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")




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
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={
                    "question": "I need to speak to a human agent right now",
                    "chat_history": chat_history,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            summary = data.get("handoff_summary", "User requested handoff.")
        except Exception:
            summary = "User requested handoff."
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
        placeholder.markdown("Thinking...")

        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]

        data = {}
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={
                    "question": prompt,
                    "chat_history": chat_history,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            final_answer = data.get("answer", "No answer found.")

            # ── Section 1: Final Answer ──
            placeholder.markdown(final_answer)

            # ── Section 2: Agent Execution timeline ──
            collected_logs = data.get("logs", [])
            if st.session_state.show_trace and collected_logs:
                timeline = _format_timeline(collected_logs)
                with st.expander("🧠 Agent Execution", expanded=False):
                    st.markdown(timeline)

            # ── Visualization ──
            viz_spec = data.get("visualization_spec")
            sql_raw = data.get("sql_result", "")
            if viz_spec and sql_raw:
                try:
                    import json
                    chart_data = json.loads(sql_raw)
                    if chart_data and isinstance(chart_data, list):
                        st.vega_lite_chart(chart_data, viz_spec, use_container_width=True)
                except Exception:
                    pass

            # ── Escalation banner ──
            if data.get("escalated"):
                st.warning(f"🤝 Escalated to human support — reason: **{data['escalation_reason']}**")
                summary = data.get("handoff_summary", "")
                if summary:
                    with st.expander("Handoff summary for agent"):
                        st.markdown(summary)

            # ── Section 3: Debug Panel ──
            if st.session_state.show_debug:
                with st.expander("Debug Info"):
                    st.json(data.get("debug", {}))

            # Cross-turn metadata accumulation deferred to follow-up proposal.

        except requests.exceptions.HTTPError as e:
            status = resp.status_code
            detail = resp.json().get("detail", str(e))
            if status == 429:
                error_msg = "Rate limit exceeded. Please wait and try again."
            else:
                error_msg = f"API Error ({status}): {detail}"
            placeholder.error(error_msg)
            final_answer = error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to API server. Make sure FastAPI is running."
            placeholder.error(error_msg)
            final_answer = error_msg
        except Exception as e:
            placeholder.error(f"Error: {e}")
            final_answer = f"Error: {e}"

        st.session_state.history.append({
            "role": "assistant",
            "content": final_answer,
            "debug": data.get("debug", {}) if st.session_state.show_debug else {},
        })
