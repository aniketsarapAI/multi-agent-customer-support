from typing import List, Literal

from pydantic import BaseModel, Field


class QueryTypeDecision(BaseModel):
    query_type: Literal["document", "database", "hybrid", "conversation"] = Field(
        ...,
        description="'document' if the question is about company policies, profile, or documents; 'database' if it requires querying structured database tables; 'hybrid' if the question has multiple parts spanning both types; 'conversation' if the user is expressing emotion, feedback, or conversational intent rather than requesting information.",
    )


class SQLRewriteDecision(BaseModel):
    refined_query: str = Field(
        ...,
        description="Rewritten SQL query intent — explicit, specific, with concrete column/table names.",
    )


class SQLQueryDecision(BaseModel):
    sql_query: str = Field(
        ...,
        description="A valid MySQL SELECT query to answer the user's question.",
    )


class SubQuestionItem(BaseModel):
    id: str = Field(..., description="Unique id like doc_0, sql_0, doc_1, etc.")
    question: str = Field(..., description="The sub-question text.")
    type: Literal["document", "database"] = Field(
        ...,
        description="'document' for company document questions, 'database' for data queries.",
    )


class DecomposeDecision(BaseModel):
    sub_questions: List[SubQuestionItem] = Field(
        ...,
        description="List of sub-questions decomposed from the compound question.",
    )


class Intent(BaseModel):
    category: Literal["information_request", "emotional", "feedback", "escalation"]
    description: str = ""


class ExecutionPlan(BaseModel):
    agents: list[Literal["rag", "sql", "conversation"]]
    parallel: bool = False
    needs_synthesis: bool = False
    priority: int = 0
    timeout_ms: int = 30000
    max_retries: int = 0
    cacheable: bool = True
    clarification: bool = False
