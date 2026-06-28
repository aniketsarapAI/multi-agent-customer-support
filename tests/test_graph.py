from unittest.mock import MagicMock

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.models.state import SupervisorState
from app.models.metadata import RAGMetadata, SQLMetadata, ConversationMetadata
from app.orchestration.registry import AgentRegistry
from app.orchestration.supervisor_react import build_react_supervisor_graph, SupervisorDecision, ToolCall
from app.pipeline.escalation import EscalationChecker


def _make_agent_result(agent: str, answer: str, success: bool = True):
    meta_cls = {"rag": RAGMetadata, "sql": SQLMetadata, "conversation": ConversationMetadata}
    return MagicMock(
        agent=agent,
        success=success,
        answer=answer,
        confidence=1.0 if success else 0.0,
        metadata=meta_cls.get(agent, RAGMetadata)(issup="fully_supported" if success else "no_support",
                                                     isuse="useful" if success else "not_useful"),
        latency_ms=10,
        logs=[f"{agent} ran"],
    )


def _make_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for name in ["rag", "sql"]:
        m = MagicMock()
        m.invoke = MagicMock(return_value=_make_agent_result(name, f"answer from {name}"))
        m.health = MagicMock(return_value=True)
        m.graph = MagicMock()
        m.capabilities = MagicMock(return_value=[name])
        registry.register(name, m)
    return registry


def _make_state(**overrides) -> SupervisorState:
    state: SupervisorState = {
        "request_id": "test-1",
        "conversation_id": "test-conv-1",
        "question": "test query",
        "original_question": "test query",
        "chat_history": [],
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
    state.update(overrides)
    return state


def _make_llm(decisions: list):
    """Create an LLM mock that returns the given SupervisorDecision objects in sequence."""
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(side_effect=decisions)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=mock_structured)
    return llm


class TestReActSupervisorGraph:
    def test_graph_compiles(self):
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=MagicMock())
        registry = _make_registry()
        graph = build_react_supervisor_graph(registry, llm, EscalationChecker(llm))
        assert graph is not None
        nodes = list(graph.get_graph().nodes.keys())
        assert "think" in nodes
        assert "route" in nodes
        assert "rag_agent" in nodes
        assert "sql_agent" in nodes
        assert "escalate" in nodes

    def test_direct_respond(self):
        """LLM responds immediately without calling any tool."""
        llm = _make_llm([
            SupervisorDecision(action="respond", answer="Hello! How can I help?"),
        ])
        registry = _make_registry()
        escalation_mock = MagicMock()
        escalation_mock.check = MagicMock(return_value=(False, "", ""))
        graph = build_react_supervisor_graph(registry, llm, escalation_mock)

        result = None
        for output in graph.stream(_make_state(question="Hi"), stream_mode="values"):
            result = output

        assert result is not None
        assert result.get("final_answer") == "Hello! How can I help?"

    def test_single_agent_respond(self):
        """LLM calls one tool, then responds."""
        llm = _make_llm([
            SupervisorDecision(action="single", tool="rag", query="Who is CEO?"),
            SupervisorDecision(action="respond", answer="The CEO is John Smith."),
        ])
        registry = _make_registry()
        escalation_mock = MagicMock()
        escalation_mock.check = MagicMock(return_value=(False, "", ""))
        graph = build_react_supervisor_graph(registry, llm, escalation_mock)

        result = None
        for output in graph.stream(_make_state(), stream_mode="values"):
            result = output

        assert result is not None
        assert result.get("final_answer") == "The CEO is John Smith."
        # Verify tool was called with correct query
        registry.get("rag").invoke.assert_called_once()
        _, kwargs = registry.get("rag").invoke.call_args
        assert kwargs["question"] == "Who is CEO?"

    def test_parallel_fan_out(self):
        """LLM calls two tools in parallel, then responds."""
        llm = _make_llm([
            SupervisorDecision(
                action="parallel",
                calls=[
                    ToolCall(tool="rag", query="Who is CEO?"),
                    ToolCall(tool="sql", query="What was revenue?"),
                ],
            ),
            SupervisorDecision(action="respond", answer="CEO is John. Revenue was $1.2M."),
        ])
        registry = _make_registry()
        escalation_mock = MagicMock()
        escalation_mock.check = MagicMock(return_value=(False, "", ""))
        graph = build_react_supervisor_graph(registry, llm, escalation_mock)

        result = None
        for output in graph.stream(_make_state(), stream_mode="values"):
            result = output

        assert result is not None
        assert result.get("final_answer") == "CEO is John. Revenue was $1.2M."
        # Both tools should have been called
        assert registry.get("rag").invoke.called
        assert registry.get("sql").invoke.called

    def test_agent_failure_triggers_retry(self):
        """Agent fails, LLM can re-decide on next iteration."""
        llm = _make_llm([
            SupervisorDecision(action="single", tool="rag", query="Who is CEO?"),
            SupervisorDecision(action="respond", answer="I could not find that information."),
        ])
        registry = _make_registry()
        # Make rag agent fail
        rag_mock = registry.get("rag")
        rag_mock.invoke = MagicMock(side_effect=RuntimeError("Service unavailable"))

        escalation_mock = MagicMock()
        escalation_mock.check = MagicMock(return_value=(False, "", ""))
        graph = build_react_supervisor_graph(registry, llm, escalation_mock)

        result = None
        for output in graph.stream(_make_state(), stream_mode="values"):
            result = output

        assert result is not None
        assert result.get("final_answer") == "I could not find that information."
        # Verify the agent was actually invoked (failed)
        assert rag_mock.invoke.called

    def test_escalation_fires(self):
        llm = _make_llm([
            SupervisorDecision(action="respond", answer="I have no idea."),
        ])
        registry = _make_registry()

        escalation_mock = MagicMock()
        escalation_mock.check = MagicMock(return_value=(True, "frustration", "Handoff summary"))
        graph = build_react_supervisor_graph(registry, llm, escalation_mock)

        result = None
        for output in graph.stream(_make_state(question="I'm angry!"), stream_mode="values"):
            result = output

        assert result is not None
        assert result.get("escalated") is True
        assert result.get("escalation_reason") == "frustration"

    def test_max_iterations_forced_respond(self):
        """When iteration_count exceeds MAX_REACT_ITERATIONS, force respond."""
        llm = _make_llm([
            SupervisorDecision(action="single", tool="rag", query="q1"),
            SupervisorDecision(action="single", tool="rag", query="q2"),
            SupervisorDecision(action="single", tool="rag", query="q3"),
            SupervisorDecision(action="single", tool="rag", query="q4"),
            SupervisorDecision(action="single", tool="rag", query="q5"),
            SupervisorDecision(action="single", tool="rag", query="q6"),  # won't be reached
        ])
        registry = _make_registry()

        escalation_mock = MagicMock()
        escalation_mock.check = MagicMock(return_value=(False, "", ""))
        graph = build_react_supervisor_graph(registry, llm, escalation_mock)

        result = None
        for output in graph.stream(_make_state(), stream_mode="values"):
            result = output

        assert result is not None
        # Should have forced a final answer after max iterations
        assert result.get("final_answer")
