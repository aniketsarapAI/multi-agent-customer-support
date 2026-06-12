from langgraph.graph import StateGraph, START, END

from app.state import State
from app.config import RECURSION_LIMIT
from app.graph.nodes import (
    decide_retrieval,
    route_after_decide,
    generate_direct,
    retrieve,
    is_relevant,
    route_after_relevance,
    generate_from_context,
    no_answer_found,
    is_sup,
    route_after_issup,
    accept_answer,
    revise_answer,
    is_use,
    route_after_isuse,
    rewrite_question,
)


def build_rag_subgraph():
    g = StateGraph(State)

    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", is_sup)
    g.add_node("revise_answer", revise_answer)
    g.add_node("is_use", is_use)
    g.add_node("rewrite_question", rewrite_question)

    g.add_edge(START, "decide_retrieval")

    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )

    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "is_relevant")

    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("no_answer_found", END)
    g.add_edge("generate_from_context", "is_sup")

    g.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {
            "accept_answer": "is_use",
            "revise_answer": "revise_answer",
        },
    )

    g.add_edge("revise_answer", "is_sup")

    g.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "END": END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("rewrite_question", "retrieve")

    subgraph = g.compile()
    subgraph.recursion_limit = RECURSION_LIMIT
    return subgraph
