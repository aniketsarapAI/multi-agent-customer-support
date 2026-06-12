import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

from app.graph.builder import build_app
from app.state import State


# ── Request / Response models ──

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[dict] = Field(default_factory=list)


limiter = Limiter(key_func=get_remote_address)


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    logs: list[str]
    debug: dict
    visualization_spec: dict | None = None
    escalated: bool = False
    escalation_reason: str = ""
    handoff_summary: str = ""
    sql_result: str = ""


# ── Graph lifecycle ──

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logging.info("Building graph (loads PDFs, builds FAISS index)...")
    _graph = build_app()
    logging.info("Graph ready")
    yield
    _graph = None


app = FastAPI(
    title="Self-RAG MCP",
    description="Multi-agent customer support with Self-RAG, SQL queries, hybrid decomposition, and human escalation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/health")
async def health():
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    return {
        "status": "healthy",
        "vector_store": "ready",
        "database": "configured",
        "llm": "configured",
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, req: ChatRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    initial_state: State = {
        "question": question,
        "original_question": question,
        "query_type": "document",
        "chat_history": req.chat_history,
        "sub_questions": [],
        "sub_results": [],
        "sub_question": None,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "need_retrieval": False,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
        "logs": [],
        "sql_query": "",
        "sql_result": "",
        "db_answer": "",
        "db_error": "",
        "visualization_spec": None,
        "escalated": False,
        "escalation_reason": "",
        "handoff_summary": "",
        "rag_docs_used": [],
        "sql_queries_executed": [],
    }

    try:
        result = None
        collected_logs: list[str] = []
        for output in _graph.stream(initial_state, stream_mode="values"):
            result = output
            new_logs = output.get("logs", [])
            if isinstance(new_logs, list):
                for log in new_logs:
                    if log not in collected_logs:
                        collected_logs.append(log)

        if result is None:
            raise HTTPException(status_code=500, detail="Graph returned no result")

        query_type = result.get("query_type", "document")
        if query_type in ("database", "hybrid"):
            answer = result.get("db_answer") or result.get("answer", "No answer found.")
        else:
            answer = result.get("answer", "No answer found.")

        debug = {
            "query_type": query_type,
            "issup": result.get("issup"),
            "isuse": result.get("isuse"),
            "use_reason": result.get("use_reason", ""),
            "evidence": result.get("evidence", []),
            "retrieved_docs": len(result.get("docs", []) or []),
            "relevant_docs": len(result.get("relevant_docs", []) or []),
            "rewrite_tries": result.get("rewrite_tries", 0),
            "retries": result.get("retries", 0),
            "sql_query": result.get("sql_query", ""),
            "sql_result": result.get("sql_result", ""),
            "db_error": result.get("db_error", ""),
            "need_retrieval": result.get("need_retrieval"),
            "sub_questions": result.get("sub_questions", []),
            "sub_results": dict(result.get("sub_results", [])),
        }

        return ChatResponse(
            answer=answer,
            query_type=query_type,
            logs=collected_logs,
            debug=debug,
            visualization_spec=result.get("visualization_spec"),
            escalated=result.get("escalated", False),
            escalation_reason=result.get("escalation_reason", ""),
            handoff_summary=result.get("handoff_summary", ""),
            sql_result=result.get("sql_result", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(e))
