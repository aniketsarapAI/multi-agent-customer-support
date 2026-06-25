import time

from app.models.agent import BaseAgent, AgentResult
from app.models.metadata import ConversationMetadata
from app.prompts import direct_generation_prompt
from app.chat_history import format_chat_history


class ConversationAgent(BaseAgent):
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, question: str, chat_history: list[dict], request_id: str) -> AgentResult:
        start = time.perf_counter()

        history_str = format_chat_history(chat_history)
        out = self._llm.invoke(
            direct_generation_prompt.format_messages(
                question=question,
                chat_history=history_str,
            )
        )
        elapsed = int((time.perf_counter() - start) * 1000)

        return AgentResult(
            agent="conversation",
            success=True,
            answer=out.content,
            confidence=1.0,
            metadata=ConversationMetadata(),
            latency_ms=elapsed,
            logs=["🔍 conversation_agent: direct response generated"],
        )

    def health(self) -> bool:
        return True

    def capabilities(self) -> list[str]:
        return ["conversation", "direct_response"]
