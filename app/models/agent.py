from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from app.models.metadata import RAGMetadata, SQLMetadata, ConversationMetadata


class AgentResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent: Literal["rag", "sql", "conversation"]
    success: bool
    answer: str
    confidence: float
    metadata: RAGMetadata | SQLMetadata | ConversationMetadata
    latency_ms: int
    logs: list[str]


class BaseAgent(Protocol):
    def invoke(self, question: str, chat_history: list[dict], request_id: str) -> AgentResult:
        ...

    def health(self) -> bool:
        ...

    def capabilities(self) -> list[str]:
        ...
