import logging

from app.config import RECURSION_LIMIT, settings
from app.infrastructure.llm import get_llm, get_grader_llm
from app.infrastructure.vector_store import create_retriever
from app.orchestration.registry import AgentRegistry
from langgraph.checkpoint.memory import MemorySaver
from app.orchestration.supervisor_react import build_react_supervisor_graph
from app.agents.rag_agent import RAGAgent
from app.agents.sql_agent import SQLAgent
from app.agents.conversation_agent import ConversationAgent
from app.services.security import SecurityPipeline
from app.services.cache import SemanticCache
from app.services.memory import InMemoryMemoryService, RedisMemoryService
from app.services.health import HealthService
from app.services.monitoring import MetricsCollector
from app.pipeline.escalation import EscalationChecker

logger = logging.getLogger(__name__)


class Application:
    """Top-level container for the multi-agent system."""

    def __init__(self, redis_url: str = "", cache_ttl: int = 600, memory_ttl: int = 86400):
        redis_url = redis_url or settings.redis_url

        # Core infrastructure
        self.llm = get_llm()
        self.retriever = create_retriever()

        # Services
        self.security = SecurityPipeline()

        # Semantic cache with optional Redis persistence
        self.cache = SemanticCache(
            redis_url=redis_url,
            ttl_seconds=cache_ttl,
            similarity_threshold=settings.semantic_cache_threshold,
            freq_questions_path=settings.freq_questions_path,
        )

        self.metrics = MetricsCollector()

        # Memory — try Redis, fallback to in-memory
        redis_memory = RedisMemoryService(redis_url, ttl_seconds=memory_ttl)
        if redis_memory.available:
            self.memory = redis_memory
            logger.info("Using Redis-backed conversation memory")
        else:
            self.memory = InMemoryMemoryService()
            logger.info("Using in-memory conversation memory (Redis unavailable)")

        # Agents
        self.registry = AgentRegistry()
        self.grader_llm = get_grader_llm()
        self.registry.register("rag", RAGAgent(self.llm, self.retriever, self.grader_llm))
        self.registry.register("sql", SQLAgent(self.llm))
        self.registry.register("conversation", ConversationAgent(self.llm))

        # Orchestration
        self.checkpointer = MemorySaver()
        self.escalation = EscalationChecker(self.llm)

        # Supervisor graph (reAct loop with parallel Send)
        self.supervisor = build_react_supervisor_graph(
            self.registry, self.llm, self.escalation, checkpointer=self.checkpointer
        )
        self.supervisor.recursion_limit = RECURSION_LIMIT

        # Health
        self.health_service = HealthService(
            registry=self.registry,
            memory=self.memory,
            cache=self.cache,
        )

    @property
    def ready(self) -> bool:
        return all([
            self.supervisor is not None,
            self.security is not None,
            self.cache is not None,
            self.metrics is not None,
        ])
