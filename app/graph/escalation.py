import logging
from typing import Literal

from langgraph.graph import END

from app.state import State
from app.models import EscalationDecision
from app.prompts import escalation_check_prompt, handoff_summary_prompt
from app.chat_history import format_chat_history
from app.email_alert import send_escalation_email

_llm = None


def set_llm(llm):
    global _llm
    _llm = llm


def check_escalation(state: State):
    logs = ["🔍 check_escalation: evaluating need for human handoff..."]
    chat_str = format_chat_history(state.get("chat_history", []))
    answer = state.get("answer") or state.get("db_answer") or ""
    issup = state.get("issup", "no_support")
    isuse = state.get("isuse", "not_useful")

    structured = _llm.with_structured_output(EscalationDecision)
    decision: EscalationDecision = structured.invoke(
        escalation_check_prompt.format_messages(
            chat_history=chat_str,
            question=state.get("original_question", state["question"]),
            answer=answer,
            issup=issup,
            isuse=isuse,
        )
    )
    reason = decision.reason if decision.escalate else ""
    logs.append(f"  escalate={decision.escalate} reason={decision.reason} issup={issup} isuse={isuse}")
    return {
        "escalated": decision.escalate,
        "escalation_reason": reason,
        "logs": logs,
    }


def generate_handoff(state: State):
    logs = ["🔍 generate_handoff: creating handoff summary for human agent..."]
    reason = state.get("escalation_reason", "")
    answer = state.get("answer") or state.get("db_answer") or ""
    sql_query = state.get("sql_query", "")
    chat_str = format_chat_history(state.get("chat_history", []))

    # Extract document titles only (no full content)
    relevant_docs = state.get("relevant_docs", []) or []
    doc_titles = []
    for d in relevant_docs:
        meta = d.metadata or {}
        title = meta.get("title") or meta.get("source", "")
        if title:
            doc_titles.append(title)
    doc_str = ", ".join(doc_titles) if doc_titles else "none"

    rag_docs = state.get("rag_docs_used", []) or []
    sql_queries = state.get("sql_queries_executed", []) or []

    out = _llm.invoke(
        handoff_summary_prompt.format_messages(
            chat_history=chat_str,
            question=state.get("original_question", state["question"]),
            answer=answer,
            reason=reason,
            current_doc_titles=doc_str,
            current_sql_query=sql_query or "none",
            rag_docs_used="\n".join(f"- {d}" for d in rag_docs) if rag_docs else "none",
            sql_queries_executed="\n".join(f"- {q}" for q in sql_queries) if sql_queries else "none",
        )
    )
    logs.append("✅ generate_handoff: done")
    try:
        send_escalation_email(
            out.content,
            reason,
            rag_docs_used=rag_docs,
            sql_queries_executed=sql_queries,
            message_count=len(state.get("chat_history", [])),
        )
        logs.append("📧 escalation email sent")
    except Exception as e:
        logs.append(f"⚠️ escalation email failed: {type(e).__name__}")
        logging.exception("Failed to send escalation email")
    return {"handoff_summary": out.content, "logs": logs}


def route_escalation(state: State) -> Literal["generate_handoff", "__end__"]:
    return "generate_handoff" if state.get("escalated") else END
