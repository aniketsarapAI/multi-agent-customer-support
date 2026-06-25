from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

from app.graph.builder import build_app
from app.state import State
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.security import SecurityPipeline
from app.cache import ResponseCache

logger = get_logger("api")


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
    security_notes: list[str] = []
    cached: bool = False
    processing_time_ms: float = 0


class HealthResponse(BaseModel):
    status: str
    graph: str
    security: str
    cache: str
    vector_store: str
    database: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: float
    avg_latency_ms: float
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float


# ── Graph lifecycle ──

_graph = None
_security = None
_cache = None
_metrics = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _security, _cache, _metrics
    logger.info("Initializing components...")
    _security = SecurityPipeline()
    _cache = ResponseCache(ttl_seconds=300)
    _metrics = MetricsCollector()
    logger.info("Building graph (loads PDFs, builds FAISS index)...")
    _graph = build_app()
    logger.info("Graph ready — all components initialized")
    yield
    _graph = None
    _security = None
    _cache = None
    _metrics = None


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
        "graph": "ready" if _graph is not None else "not_ready",
        "security": "ready" if _security is not None else "not_ready",
        "cache": "ready" if _cache is not None else "not_ready",
        "vector_store": "ready",
        "database": "configured",
    }


@app.get("/metrics")
async def metrics():
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not ready")
    return _metrics.summary


@app.get("/cache/stats")
async def cache_stats():
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not ready")
    return _cache.stats


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, req: ChatRequest):
    if _graph is None or _security is None or _cache is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    timer = RequestTimer()

    is_allowed, cleaned_question, security_notes = _security.check_input(question)
    if not is_allowed:
        logger.warning("Security check blocked input", extra={"extra_data": {"question": question[:100]}})
        raise HTTPException(status_code=400, detail=security_notes[0] if security_notes else "Input blocked by security check")

    if cleaned_question != question:
        logger.info("Input cleaned by security pipeline", extra={"extra_data": {"original": question[:50], "cleaned": cleaned_question[:50]}})

    cached_response = _cache.get(cleaned_question)
    if cached_response is not None:
        logger.info("Cache hit", extra={"extra_data": {"question": cleaned_question[:50]}})
        timer.elapsed_ms = 0
        _metrics.record_request(latency_ms=0, cache_hit=True)
        return ChatResponse(
            answer=cached_response,
            query_type="cached",
            logs=["📦 Response served from cache"],
            debug={},
            security_notes=security_notes,
            cached=True,
            processing_time_ms=0,
        )

    initial_state: State = {
        "question": cleaned_question,
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

    timer.__enter__()

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

        safe_answer, output_warnings = _security.check_output(answer)
        if output_warnings:
            logger.warning("Output validation warnings", extra={"extra_data": {"warnings": output_warnings}})

        _cache.set(cleaned_question, safe_answer)

        timer.__exit__(None, None, None)
        _metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=False,
            tokens_input=0,
            tokens_output=0,
            cache_hit=False,
        )

        return ChatResponse(
            answer=safe_answer,
            query_type=query_type,
            logs=collected_logs,
            debug=debug,
            visualization_spec=result.get("visualization_spec"),
            escalated=result.get("escalated", False),
            escalation_reason=result.get("escalation_reason", ""),
            handoff_summary=result.get("handoff_summary", ""),
            sql_result=result.get("sql_result", ""),
            security_notes=security_notes + output_warnings,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
        )

    except HTTPException:
        timer.__exit__(None, None, None)
        _metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
        raise
    except Exception as e:
        timer.__exit__(None, None, None)
        logger.exception("Chat request failed", extra={"extra_data": {"question": req.question[:100]}})
        _metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
        raise HTTPException(status_code=500, detail=str(e))
