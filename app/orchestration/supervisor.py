from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.models.state import SupervisorState
from app.models.metadata import SQLMetadata
from app.models.agent import AgentResult
from app.orchestration.registry import AgentRegistry
from app.prompts import synthesise_hybrid_prompt


def build_supervisor_graph(registry: AgentRegistry, llm):
    g = StateGraph(SupervisorState)

    def delegate(state: SupervisorState):
        logs = [f"🔍 supervisor: delegating to {state['execution_plan'].agents}"]
        return {"logs": logs}

    def route_to_agents(state: SupervisorState) -> list[Send]:
        plan = state["execution_plan"]
        sends = []
        for agent_name in plan.agents:
            target = {"rag": "rag_agent", "sql": "sql_agent", "conversation": "conversation_agent"}[agent_name]
            sends.append(Send(target, state))
        return sends

    def make_agent_node(agent_name: str):
        def agent_node(state: SupervisorState):
            agent = registry.get(agent_name)
            result = agent.invoke(
                question=state["question"],
                chat_history=state["chat_history"],
                request_id=state["request_id"],
            )
            return {"agent_results": [result], "logs": [f"✅ supervisor: {agent_name} agent completed"]}
        return agent_node

    def synthesise(state: SupervisorState):
        logs = ["🔍 supervisor: synthesising results..."]
        plan = state["execution_plan"]
        results: list[AgentResult] = state.get("agent_results", [])

        if not results:
            logs.append("⚠️ supervisor: no agent results")
            return {"final_answer": "No answer found.", "logs": logs}

        final = results[0].answer
        viz = None

        if plan.needs_synthesis and len(results) > 1:
            partials = "\n\n".join(f"{r.agent}: {r.answer}" for r in results)
            out = llm.invoke(
                synthesise_hybrid_prompt.format_messages(
                    question=state["question"],
                    partial_answers=partials,
                )
            )
            final = out.content
            logs.append("✅ supervisor: hybrid synthesis done")
        elif len(results) == 1:
            # Single agent — pass through
            pass

        # Extract visualization if any SQL result had one
        for r in results:
            if isinstance(r.metadata, SQLMetadata) and r.metadata.visualization_spec:
                viz = r.metadata.visualization_spec
                break

        logs.append(f"✅ supervisor: final answer ready ({len(final)} chars)")
        return {"final_answer": final, "visualization_spec": viz, "logs": logs}

    g.add_node("delegate", delegate)
    g.add_node("rag_agent", make_agent_node("rag"))
    g.add_node("sql_agent", make_agent_node("sql"))
    g.add_node("conversation_agent", make_agent_node("conversation"))
    g.add_node("synthesise", synthesise)

    g.add_edge(START, "delegate")

    g.add_conditional_edges(
        "delegate",
        route_to_agents,
        ["rag_agent", "sql_agent", "conversation_agent"],
    )

    g.add_edge("rag_agent", "synthesise")
    g.add_edge("sql_agent", "synthesise")
    g.add_edge("conversation_agent", "synthesise")
    g.add_edge("synthesise", END)

    return g.compile()
