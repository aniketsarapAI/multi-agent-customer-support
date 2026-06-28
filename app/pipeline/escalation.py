import logging
from typing import Literal

from pydantic import BaseModel

from app.prompts import escalation_check_prompt, handoff_summary_prompt
from app.chat_history import format_chat_history
from app.infrastructure.email_alert import send_escalation_email


class EscalationDecision(BaseModel):
    escalate: bool
    reason: Literal[
        "human_requested",
        "complaint",
        "frustration",
        "repeated_negative_sentiment",
        "unresolved_issue",
        "none",
    ]


logger = logging.getLogger(__name__)


class EscalationChecker:
    def __init__(self, llm):
        self._llm = llm

    def check(self, question: str, answer: str, chat_history: list[dict], issup: str = "", isuse: str = "") -> tuple[bool, str, str]:
        chat_str = format_chat_history(chat_history)
        structured = self._llm.with_structured_output(
            EscalationDecision, default=lambda: EscalationDecision(escalate=False, reason="none")
        )
        decision: EscalationDecision = structured.invoke(
            escalation_check_prompt.format_messages(
                chat_history=chat_str,
                question=question,
                answer=answer,
                issup=issup,
                isuse=isuse,
            )
        )

        handoff = ""
        if decision.escalate:
            handoff = self._generate_handoff(question, answer, chat_history, decision.reason)
            try:
                send_escalation_email(handoff, decision.reason, message_count=len(chat_history))
            except Exception:
                logger.exception("Failed to send escalation email")

        return decision.escalate, decision.reason, handoff

    def _generate_handoff(self, question: str, answer: str, chat_history: list[dict], reason: str) -> str:
        chat_str = format_chat_history(chat_history)
        out = self._llm.invoke(
            handoff_summary_prompt.format_messages(
                chat_history=chat_str,
                question=question,
                answer=answer,
                reason=reason,
                current_doc_titles="none",
                current_sql_query="none",
                rag_docs_used="none",
                sql_queries_executed="none",
            )
        )
        return out.content
