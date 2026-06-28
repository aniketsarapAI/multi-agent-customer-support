from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

from app.graph.builder import Application
from app.models.state import SupervisorState
from app.services.monitoring import get_logger, RequestTimer

logger = get_logger("api")


# ── Request / Response models ──

class ChatRequest(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
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


# ── Application lifecycle ──

_app: Application | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app
    logger.info("Initializing multi-agent application...")
    _app = Application()
    logger.info("Application ready — all agents, services, and graph initialized")
    yield
    _app = None


app = FastAPI(
    title="Self-RAG MCP",
    description="Multi-agent customer support with Self-RAG, SQL queries, hybrid decomposition, and human escalation.",
    version="2.0.0",
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
    if _app is None:
        raise HTTPException(status_code=503, detail="Application not ready")
    result = _app.health_service.check()
    return HealthResponse(
        status=result.get("status", "unknown"),
        graph="ready" if _app.supervisor is not None else "not_ready",
        security="ready" if _app.security is not None else "not_ready",
        cache="ready" if _app.cache is not None else "not_ready",
        vector_store="ready",
        database="configured",
    )


@app.get("/metrics")
async def metrics():
    if _app is None or _app.metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not ready")
    return _app.metrics.summary


@app.get("/cache/stats")
async def cache_stats():
    if _app is None or _app.cache is None:
        raise HTTPException(status_code=503, detail="Cache not ready")
    return _app.cache.stats


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, req: ChatRequest):
    if _app is None or not _app.ready:
        raise HTTPException(status_code=503, detail="Server not ready")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    timer = RequestTimer()

    # ── Security check ──
    is_allowed, cleaned_question, security_notes = _app.security.check_input(question)
    if not is_allowed:
        logger.warning("Security check blocked input", extra={"extra_data": {"question": question[:100]}})
        raise HTTPException(status_code=400, detail=security_notes[0] if security_notes else "Input blocked by security check")

    if cleaned_question != question:
        logger.info("Input cleaned by security pipeline", extra={"extra_data": {"original": question[:50], "cleaned": cleaned_question[:50]}})

    # ── Cache check ──
    cached_response = _app.cache.get(cleaned_question)
    if cached_response is not None:
        logger.info("Cache hit", extra={"extra_data": {"question": cleaned_question[:50]}})
        timer.elapsed_ms = 0
        _app.metrics.record_request(latency_ms=0, cache_hit=True)
        return ChatResponse(
            answer=cached_response,
            query_type="cached",
            logs=["📦 Response served from cache"],
            debug={},
            security_notes=security_notes,
            cached=True,
            processing_time_ms=0,
        )

    # ── Load conversation memory ──
    request_id = str(uuid4())
    conversation_id = req.conversation_id or str(uuid4())
    saved_state = _app.memory.load(conversation_id)
    saved_history = saved_state.get("chat_history", [])
    # Prefer the longer history (handles client-side truncation)
    merged_history = req.chat_history if len(req.chat_history) >= len(saved_history) else saved_history

    logger.info(
        "Processing request",
        extra={"extra_data": {"request_id": request_id, "question": cleaned_question[:50]}},
    )

    # ── Execute supervisor graph ──
    initial_state: SupervisorState = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "question": cleaned_question,
        "original_question": question,
        "chat_history": merged_history,
        "messages": [],
        "supervisor_decision": None,
        "final_answer": "",
        "logs": [],
        "visualization_spec": None,
        "escalated": False,
        "escalation_reason": "",
        "handoff_summary": "",
        "iteration_count": 0,
    }

    timer.__enter__()
    collected_logs: list[str] = []

    try:
        result = None
        config = {"configurable": {"thread_id": conversation_id}}
        for output in _app.supervisor.stream(initial_state, config=config, stream_mode="values"):
            result = output
            new_logs = output.get("logs", [])
            if isinstance(new_logs, list):
                for log in new_logs:
                    if log not in collected_logs:
                        collected_logs.append(log)

        if result is None:
            raise HTTPException(status_code=500, detail="Graph returned no result")

        final_answer = result.get("final_answer", "No answer found.")

        # ── Security output check (escalation is handled in-graph) ──
        safe_answer, output_warnings = _app.security.check_output(final_answer)
        security_notes.extend(output_warnings)

        # ── Read escalation results from graph state ──
        escalated = result.get("escalated", False)
        escalation_reason = result.get("escalation_reason", "")
        handoff_summary = result.get("handoff_summary", "")
        visualization_spec = result.get("visualization_spec")

        # ── Build debug info from messages ──
        messages = result.get("messages", [])
        tools_used = sorted(set(m["tool"] for m in messages if m.get("role") == "tool"))
        query_type = "+".join(tools_used) if tools_used else "direct"

        issup = ""
        isuse = ""
        for msg in messages:
            metadata = msg.get("metadata", {})
            if metadata.get("issup"):
                issup = metadata["issup"]
            if metadata.get("isuse"):
                isuse = metadata["isuse"]

        sql_result_str = ""
        for msg in messages:
            metadata = msg.get("metadata", {})
            if metadata.get("sql_result"):
                sql_result_str = metadata["sql_result"]

        debug = {
            "query_type": query_type,
            "tools_used": tools_used,
            "issup": issup,
            "isuse": isuse,
            "agent_count": len(tools_used),
        }

        # ── Cache ──
        _app.cache.set(cleaned_question, safe_answer)

        timer.__exit__(None, None, None)
        _app.metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=False,
            cache_hit=False,
        )

        return ChatResponse(
            answer=safe_answer,
            query_type=query_type,
            logs=collected_logs,
            debug=debug,
            visualization_spec=visualization_spec,
            escalated=escalated,
            escalation_reason=escalation_reason,
            handoff_summary=handoff_summary,
            sql_result=sql_result_str,
            security_notes=security_notes,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
        )

    except HTTPException:
        timer.__exit__(None, None, None)
        _app.metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
        raise
    except Exception as e:
        timer.__exit__(None, None, None)
        logger.exception("Chat request failed", extra={"extra_data": {"question": req.question[:100]}})
        _app.metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
        raise HTTPException(status_code=500, detail=str(e))
