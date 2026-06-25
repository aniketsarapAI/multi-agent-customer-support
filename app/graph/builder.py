from app.config import RECURSION_LIMIT
from app.infrastructure.llm import get_llm
from app.infrastructure.vector_store import create_retriever
from app.orchestration.registry import AgentRegistry
from app.orchestration.planner import Planner
from app.orchestration.supervisor import build_supervisor_graph
from app.agents.rag_agent import RAGAgent
from app.agents.sql_agent import SQLAgent
from app.agents.conversation_agent import ConversationAgent
from app.services.security import SecurityPipeline
from app.services.cache import ResponseCache
from app.services.memory import InMemoryMemoryService
from app.services.health import HealthService
from app.services.monitoring import MetricsCollector
from app.pipeline.escalation import EscalationChecker
from app.pipeline.post_processing import PostProcessingPipeline


class Application:
    """Top-level container for the multi-agent system."""

    def __init__(self):
        # Core infrastructure
        self.llm = get_llm()
        self.retriever = create_retriever()

        # Services
        self.security = SecurityPipeline()
        self.cache = ResponseCache(ttl_seconds=300)
        self.metrics = MetricsCollector()
        self.memory = InMemoryMemoryService()

        # Agents
        self.registry = AgentRegistry()
        self.registry.register("rag", RAGAgent(self.llm, self.retriever))
        self.registry.register("sql", SQLAgent(self.llm))
        self.registry.register("conversation", ConversationAgent(self.llm))

        # Orchestration
        self.planner = Planner(self.llm)

        # Supervisor graph (compiled)
        self.supervisor = build_supervisor_graph(self.registry, self.llm)
        self.supervisor.recursion_limit = RECURSION_LIMIT

        # Pipeline
        self.escalation = EscalationChecker(self.llm)
        self.post_processing = PostProcessingPipeline(self.security, self.escalation)

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
