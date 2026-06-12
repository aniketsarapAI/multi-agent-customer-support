from typing import List, Literal
from pydantic import BaseModel, Field


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


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False.",
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that can directly answer the question.",
    )


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs.",
    )


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
