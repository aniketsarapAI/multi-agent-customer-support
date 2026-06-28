from pydantic import BaseModel, Field


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
