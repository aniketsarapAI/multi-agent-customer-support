from langgraph.graph import StateGraph, START, END

from app.state import State
from app.llm import get_llm
from app.vector_store import create_retriever
from app.config import RECURSION_LIMIT
from app.graph.nodes import (
    set_retriever as set_rag_retriever,
    set_llm as set_rag_llm,
    generate_direct,
)
from app.graph.rag_subgraph import build_rag_subgraph
from app.graph.escalation import (
    set_llm as set_esc_llm,
    check_escalation,
    generate_handoff,
    route_escalation,
)
from app.graph.sql_nodes import (
    set_llm as set_sql_llm,
    classify_question,
    route_after_classify,
    rewrite_sql_query,
    decompose_question,
    route_sub_questions,
    run_rag_sub,
    run_sql_sub,
    synthesise_hybrid,
    generate_sql,
    execute_sql_node,
    visualize_sql_result,
    summarize_sql_result,
)


def build_app():
    llm = get_llm()
    retriever = create_retriever()

    set_rag_llm(llm)
    set_rag_retriever(retriever)
    set_sql_llm(llm)
    set_esc_llm(llm)

    rag_subgraph = build_rag_subgraph()

    g = StateGraph(State)

    # ── classify ──
    g.add_node("classify_question", classify_question)

    # ── RAG path (compiled subgraph) ──
    g.add_node("rag_pipeline", rag_subgraph)

    # ── SQL path (linear, single question) ──
    g.add_node("rewrite_sql_query", rewrite_sql_query)
    g.add_node("generate_sql", generate_sql)
    g.add_node("execute_sql", execute_sql_node)
    g.add_node("visualize_sql", visualize_sql_result)
    g.add_node("summarize_sql", summarize_sql_result)

    # ── Hybrid path (fan-out / fan-in) ──
    g.add_node("decompose_question", decompose_question)
    g.add_node("run_rag_sub", run_rag_sub)
    g.add_node("run_sql_sub", run_sql_sub)
    g.add_node("synthesise_hybrid", synthesise_hybrid)

    # ── Conversation path (direct answer, no retrieval) ──
    g.add_node("generate_direct", generate_direct)

    # ── Top-level edges ──
    g.add_edge(START, "classify_question")

    g.add_conditional_edges(
        "classify_question",
        route_after_classify,
        {
            "retrieve": "rag_pipeline",
            "rewrite_sql_query": "rewrite_sql_query",
            "decompose_question": "decompose_question",
            "generate_direct": "generate_direct",
        },
    )

    g.add_edge("rag_pipeline", "check_escalation")
    g.add_edge("generate_direct", "check_escalation")

    g.add_edge("rewrite_sql_query", "generate_sql")
    g.add_edge("generate_sql", "execute_sql")
    g.add_edge("execute_sql", "visualize_sql")
    g.add_edge("visualize_sql", "summarize_sql")
    g.add_edge("summarize_sql", "check_escalation")

    # ── Hybrid fan-out ──
    g.add_conditional_edges(
        "decompose_question",
        route_sub_questions,
        ["run_rag_sub", "run_sql_sub"],
    )

    # ── Hybrid fan-in ──
    g.add_edge("run_rag_sub", "synthesise_hybrid")
    g.add_edge("run_sql_sub", "synthesise_hybrid")
    g.add_edge("synthesise_hybrid", "check_escalation")

    # ── Escalation check (runs after every answer) ──
    g.add_node("check_escalation", check_escalation)
    g.add_node("generate_handoff", generate_handoff)

    g.add_conditional_edges(
        "check_escalation",
        route_escalation,
        {"generate_handoff": "generate_handoff", END: END},
    )
    g.add_edge("generate_handoff", END)

    app = g.compile()
    app.recursion_limit = RECURSION_LIMIT
    return app
