from typing import Annotated, List, Optional, Sequence, TypedDict, Literal
import operator

from langchain_core.documents import Document


class State(TypedDict):
    question: str
    original_question: str
    query_type: Literal["document", "database", "hybrid", "conversation"]

    chat_history: Annotated[list[dict], operator.add]

    sub_questions: List[dict]
    sub_results: Annotated[Sequence[tuple[str, str]], operator.add]
    sub_question: Optional[dict]

    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str

    logs: Annotated[list[str], operator.add]

    sql_query: str
    sql_result: str
    db_answer: str
    db_error: str
    visualization_spec: Optional[dict]

    escalated: bool
    escalation_reason: str
    handoff_summary: str

    rag_docs_used: list[str]
    sql_queries_executed: list[str]
