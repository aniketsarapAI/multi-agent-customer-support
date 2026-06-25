from pydantic import BaseModel

from app.services.security import SecurityPipeline
from app.pipeline.escalation import EscalationChecker


class PipelineResult(BaseModel):
    answer: str
    escalated: bool
    escalation_reason: str
    handoff_summary: str
    security_notes: list[str]


class PostProcessingPipeline:
    def __init__(self, security: SecurityPipeline, escalation: EscalationChecker):
        self._security = security
        self._escalation = escalation

    def run(
        self,
        final_answer: str,
        question: str,
        chat_history: list[dict],
        issup: str = "",
        isuse: str = "",
    ) -> PipelineResult:
        security_notes: list[str] = []

        safe_answer, output_warnings = self._security.check_output(final_answer)
        security_notes.extend(output_warnings)

        escalated = False
        escalation_reason = ""
        handoff_summary = ""

        if not escalated:  # Only auto-check if not already escalated
            escalated, escalation_reason, handoff_summary = self._escalation.check(
                question=question,
                answer=safe_answer,
                chat_history=chat_history,
                issup=issup,
                isuse=isuse,
            )

        return PipelineResult(
            answer=safe_answer,
            escalated=escalated,
            escalation_reason=escalation_reason,
            handoff_summary=handoff_summary,
            security_notes=security_notes,
        )
