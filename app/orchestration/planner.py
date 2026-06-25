from app.models.planning import Intent, ExecutionPlan, QueryTypeDecision
from app.prompts import classify_question_prompt_v2
from app.chat_history import format_chat_history


class Planner:
    def __init__(self, llm):
        self._llm = llm

    def run(self, question: str, chat_history: list[dict]) -> tuple[Intent, ExecutionPlan]:
        intent = self._classify_intent(question, chat_history)
        plan = self._build_plan(intent, question)
        return intent, plan

    def _classify_intent(self, question: str, chat_history: list[dict]) -> Intent:
        chat_str = format_chat_history(chat_history)
        structured = self._llm.with_structured_output(QueryTypeDecision)
        decision: QueryTypeDecision = structured.invoke(
            classify_question_prompt_v2.format_messages(
                question=question,
                chat_history=chat_str,
            )
        )
        mapping = {
            "document": "information_request",
            "database": "information_request",
            "hybrid": "information_request",
            "conversation": "emotional",
        }
        return Intent(category=mapping.get(decision.query_type, "information_request"), description=decision.query_type)

    def _build_plan(self, intent: Intent, question: str) -> ExecutionPlan:
        qt = intent.description
        if qt == "database":
            return ExecutionPlan(agents=["sql"], cacheable=True)
        elif qt == "hybrid":
            return ExecutionPlan(agents=["rag", "sql"], parallel=True, needs_synthesis=True, cacheable=False, timeout_ms=60000)
        elif qt == "conversation":
            return ExecutionPlan(agents=["conversation"], cacheable=False)
        else:
            return ExecutionPlan(agents=["rag"], cacheable=True)
