from typing import Annotated, Optional, TypedDict
import operator

from app.models.agent import AgentResult
from app.models.planning import ExecutionPlan


class SupervisorState(TypedDict):
    request_id: str
    conversation_id: str
    question: str
    original_question: str
    chat_history: list[dict]
    execution_plan: Optional[ExecutionPlan]
    agent_results: Annotated[list[AgentResult], operator.add]
    final_answer: str
    logs: Annotated[list[str], operator.add]
    visualization_spec: dict | None


class RAGAgentState(TypedDict):
    request_id: str
    question: str
    answer: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: list
    relevant_docs: list
    context: str
    issup: str
    evidence: list[str]
    retries: int
    isuse: str
    use_reason: str
    logs: Annotated[list[str], operator.add]


class SQLAgentState(TypedDict):
    request_id: str
    question: str
    answer: str
    sql_query: str
    sql_result: str
    db_error: str
    visualization_spec: dict | None
    logs: Annotated[list[str], operator.add]


class ConversationAgentState(TypedDict):
    request_id: str
    question: str
    answer: str
    chat_history: list[dict]
    logs: Annotated[list[str], operator.add]
