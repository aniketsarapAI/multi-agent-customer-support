from typing import Annotated, Optional, TypedDict
import operator


class SupervisorState(TypedDict):
    request_id: str
    conversation_id: str
    question: str
    original_question: str
    chat_history: list[dict]
    messages: Annotated[list[dict], operator.add]
    supervisor_decision: Optional[dict]
    final_answer: str
    logs: Annotated[list[str], operator.add]
    visualization_spec: dict | None
    escalated: bool
    escalation_reason: str
    handoff_summary: str
    iteration_count: int


class RAGAgentState(TypedDict):
    request_id: str
    question: str
    chat_history: list[dict]
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
    original_question: str
    chat_history: list[dict]
    answer: str
    sql_query: str
    sql_result: str
    db_error: str
    visualization_spec: dict | None
    issup: str
    isuse: str
    use_reason: str
    retry_count: int
    logs: Annotated[list[str], operator.add]


class ConversationAgentState(TypedDict):
    request_id: str
    question: str
    answer: str
    chat_history: list[dict]
    logs: Annotated[list[str], operator.add]
