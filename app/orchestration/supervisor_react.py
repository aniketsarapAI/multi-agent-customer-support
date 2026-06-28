import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.models.state import SupervisorState
from app.models.metadata import SQLMetadata
from app.orchestration.registry import AgentRegistry
from app.pipeline.escalation import EscalationChecker
from app.chat_history import format_chat_history
from app.prompts import react_supervisor_prompt

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 5

_AGENT_TARGET = {
    "rag": "rag_agent",
    "sql": "sql_agent",
    "conversation": "conversation_agent",
}


class ToolCall(BaseModel):
    tool: Literal["rag", "sql", "conversation"]
    query: str = Field(description="The specific question to ask this tool")


class SupervisorDecision(BaseModel):
    action: Literal["single", "parallel", "respond"]
    tool: Optional[Literal["rag", "sql", "conversation"]] = Field(None, description="Required if action='single'")
    query: Optional[str] = Field(None, description="Required if action='single'")
    calls: Optional[list[ToolCall]] = Field(None, description="Required if action='parallel'")
    answer: Optional[str] = Field(None, description="Required if action='respond'")


def format_tool_results(messages: list[dict]) -> str:
    if not messages:
        return "None yet."
    parts = []
    for msg in messages:
        if msg.get("role") == "tool":
            status = "Success" if msg.get("success") else "Failed"
            parts.append(
                f"Tool: {msg['tool']}\n"
                f"Query: {msg['query']}\n"
                f"Result: {msg['content']}\n"
                f"Status: {status}"
            )
    return "\n\n---\n\n".join(parts) if parts else "None yet."


def build_react_supervisor_graph(registry: AgentRegistry, llm, escalation_checker: EscalationChecker, checkpointer=None):
    g = StateGraph(SupervisorState)

    def think(state: SupervisorState):
        logs = []
        iteration = state.get("iteration_count", 0) + 1
        logs.append(f"🤔 reAct: thinking (iteration {iteration})...")

        if iteration > MAX_REACT_ITERATIONS:
            logs.append("⚠️ reAct: max iterations reached — forcing respond")
            fallback = "I was unable to fully answer your question."
            return {
                "supervisor_decision": {"action": "respond", "answer": fallback},
                "final_answer": fallback,
                "iteration_count": iteration,
                "logs": logs,
            }

        tool_results_str = format_tool_results(state.get("messages", []))

        structured = llm.with_structured_output(
            SupervisorDecision,
            default=lambda: SupervisorDecision(action="respond", answer="I apologize, I encountered an error processing your request.")
        )
        decision: SupervisorDecision = structured.invoke(
            react_supervisor_prompt.format_messages(
                chat_history=format_chat_history(state.get("chat_history", [])),
                question=state["question"],
                tool_results=tool_results_str,
            )
        )

        if decision.action == "respond":
            logs.append("✅ reAct: answering directly")
            return {
                "supervisor_decision": decision.model_dump(),
                "final_answer": decision.answer or "",
                "iteration_count": iteration,
                "logs": logs,
            }

        logs.append(f"✅ reAct: → {decision.action}")
        return {
            "supervisor_decision": decision.model_dump(),
            "iteration_count": iteration,
            "logs": logs,
        }

    def route_think(state: SupervisorState):
        return state["supervisor_decision"]["action"]

    def route_agents(state: SupervisorState) -> list[Send]:
        decision = state["supervisor_decision"]
        action = decision.get("action", "single")

        if action == "single":
            tool = decision.get("tool", "rag")
            target = _AGENT_TARGET.get(tool)
            if target is None:
                return []
            sub_state = dict(state)
            sub_state["question"] = decision.get("query", state["question"])
            return [Send(target, sub_state)]

        sends = []
        for call in decision.get("calls", []):
            target = _AGENT_TARGET.get(call["tool"])
            if target is None:
                continue
            sub_state = dict(state)
            sub_state["question"] = call["query"]
            sends.append(Send(target, sub_state))
        return sends

    _AGENT_KEY = {"rag_agent": "rag", "sql_agent": "sql", "conversation_agent": "conversation"}

    def make_agent_node(agent_name: str):
        def agent_node(state: SupervisorState):
            decision = state.get("supervisor_decision", {})
            query = decision.get("query", state["question"])
            agent = registry.get(_AGENT_KEY.get(agent_name, agent_name))
            try:
                result = agent.invoke(
                    question=query,
                    chat_history=state["chat_history"],
                    request_id=state["request_id"],
                )
                msg = {
                    "role": "tool",
                    "tool": agent_name,
                    "query": query,
                    "content": result.answer,
                    "success": result.success,
                    "metadata": result.metadata.model_dump() if result.metadata else {},
                }
                ret = {
                    "messages": [msg],
                    "logs": [f"✅ reAct: {_AGENT_KEY.get(agent_name, agent_name)} completed"],
                }
                if isinstance(result.metadata, SQLMetadata) and result.metadata.visualization_spec:
                    ret["visualization_spec"] = result.metadata.visualization_spec
                return ret
            except Exception as e:
                agent_key = _AGENT_KEY.get(agent_name, agent_name)
                logger.error(f"Agent {agent_key} failed", exc_info=True)
                msg = {
                    "role": "tool",
                    "tool": agent_key,
                    "query": query,
                    "content": "I encountered an issue processing this request.",
                    "success": False,
                    "metadata": {},
                }
                return {
                    "messages": [msg],
                    "logs": [f"❌ reAct: {_AGENT_KEY.get(agent_name, agent_name)} failed — {str(e)}"],
                }
        return agent_node

    def escalate(state: SupervisorState):
        logs = ["🔍 reAct: checking escalation..."]
        final_answer = state.get("final_answer", "")

        issup = ""
        isuse = ""
        for msg in state.get("messages", []):
            metadata = msg.get("metadata", {})
            ar_issup = metadata.get("issup", "")
            ar_isuse = metadata.get("isuse", "")
            if ar_issup:
                issup = ar_issup
            if ar_isuse:
                isuse = ar_isuse

        escalated, reason, handoff = escalation_checker.check(
            question=state["question"],
            answer=final_answer,
            chat_history=state.get("chat_history", []),
            issup=issup,
            isuse=isuse,
        )
        logs.append(f"{'🚨' if escalated else '✅'} reAct: escalation={escalated}, reason={reason}")
        return {
            "escalated": escalated,
            "escalation_reason": reason if escalated else "",
            "handoff_summary": handoff if escalated else "",
            "logs": logs,
        }

    g.add_node("think", think)
    g.add_node("route", lambda s: {"logs": []})
    for name in _AGENT_TARGET.values():
        g.add_node(name, make_agent_node(name))
    g.add_node("escalate", escalate)

    g.add_edge(START, "think")

    g.add_conditional_edges(
        "think",
        route_think,
        {
            "single": "route",
            "parallel": "route",
            "respond": "escalate",
        },
    )

    g.add_conditional_edges(
        "route",
        route_agents,
        list(_AGENT_TARGET.values()),
    )

    for name in _AGENT_TARGET.values():
        g.add_edge(name, "think")

    g.add_edge("escalate", END)

    compiled = g.compile(checkpointer=checkpointer)
    compiled.recursion_limit = 50
    return compiled
