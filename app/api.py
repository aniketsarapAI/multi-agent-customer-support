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
from app.models.planning import ExecutionPlan
from app.models.metadata import SQLMetadata
from app.monitoring import get_logger, RequestTimer

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

    # ── Plan ──
    request_id = str(uuid4())
    conversation_id = req.conversation_id or str(uuid4())
    intent, execution_plan = _app.planner.run(cleaned_question, req.chat_history)

    logger.info(
        "Planner result",
        extra={"extra_data": {"request_id": request_id, "intent": intent.category, "plan": execution_plan.agents}},
    )

    # ── Execute supervisor graph ──
    initial_state: SupervisorState = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "question": cleaned_question,
        "original_question": question,
        "chat_history": req.chat_history,
        "execution_plan": execution_plan,
        "agent_results": [],
        "final_answer": "",
        "logs": [],
        "visualization_spec": None,
    }

    timer.__enter__()
    collected_logs: list[str] = []

    try:
        result = None
        for output in _app.supervisor.stream(initial_state, stream_mode="values"):
            result = output
            new_logs = output.get("logs", [])
            if isinstance(new_logs, list):
                for log in new_logs:
                    if log not in collected_logs:
                        collected_logs.append(log)

        if result is None:
            raise HTTPException(status_code=500, detail="Graph returned no result")

        final_answer = result.get("final_answer", "No answer found.")
        agent_results = result.get("agent_results", [])

        # ── Post-processing ──
        issup = ""
        isuse = ""
        for ar in agent_results:
            if hasattr(ar, "metadata") and ar.metadata:
                issup = getattr(ar.metadata, "issup", "")
                isuse = getattr(ar.metadata, "isuse", "")
                break

        pipeline_result = _app.post_processing.run(
            final_answer=final_answer,
            question=question,
            chat_history=req.chat_history,
            issup=issup,
            isuse=isuse,
        )

        # ── Build debug info ──
        query_type = "+".join(execution_plan.agents) if execution_plan.needs_synthesis else execution_plan.agents[0]

        sql_result_str = ""
        visualization_spec = None
        for ar in agent_results:
            if isinstance(ar.metadata, SQLMetadata):
                if ar.metadata.sql_result:
                    sql_result_str = ar.metadata.sql_result
                if ar.metadata.visualization_spec:
                    visualization_spec = ar.metadata.visualization_spec

        debug = {
            "query_type": query_type,
            "intent": intent.category,
            "execution_plan": execution_plan.agents,
            "issup": issup,
            "isuse": isuse,
            "agent_count": len(agent_results),
        }

        # ── Cache ──
        _app.cache.set(cleaned_question, pipeline_result.answer)

        timer.__exit__(None, None, None)
        _app.metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=False,
            cache_hit=False,
        )

        return ChatResponse(
            answer=pipeline_result.answer,
            query_type=query_type,
            logs=collected_logs,
            debug=debug,
            visualization_spec=visualization_spec,
            escalated=pipeline_result.escalated,
            escalation_reason=pipeline_result.escalation_reason,
            handoff_summary=pipeline_result.handoff_summary,
            sql_result=sql_result_str,
            security_notes=security_notes + pipeline_result.security_notes,
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
