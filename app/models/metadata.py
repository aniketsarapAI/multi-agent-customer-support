from pydantic import BaseModel, Field


class RAGMetadata(BaseModel):
    retrieved_docs: int = 0
    relevant_docs: int = 0
    issup: str = ""
    isuse: str = ""
    use_reason: str = ""
    rewrite_tries: int = 0
    retries: int = 0


class SQLMetadata(BaseModel):
    sql_query: str = ""
    sql_result: str = ""
    db_error: str = ""
    visualization_spec: dict | None = None
    issup: str = ""
    isuse: str = ""
    use_reason: str = ""


class ConversationMetadata(BaseModel):
    pass


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
    issup: str
    evidence: list[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: str
    reason: str = Field(..., description="Short reason in 1 line.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs.",
    )
