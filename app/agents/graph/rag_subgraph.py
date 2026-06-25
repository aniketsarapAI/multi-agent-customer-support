from langgraph.graph import StateGraph, START, END

from app.models.state import RAGAgentState
from app.config import RECURSION_LIMIT
from app.agents.graph.nodes import (
    make_decide_retrieval,
    route_after_decide,
    make_generate_direct,
    make_retrieve,
    make_is_relevant,
    route_after_relevance,
    make_generate_from_context,
    no_answer_found,
    make_is_sup,
    route_after_issup,
    accept_answer,
    make_revise_answer,
    make_is_use,
    route_after_isuse,
    make_rewrite_question,
)


def build_rag_subgraph(llm, retriever):
    g = StateGraph(RAGAgentState)

    g.add_node("decide_retrieval", make_decide_retrieval(llm))
    g.add_node("generate_direct", make_generate_direct(llm))
    g.add_node("retrieve", make_retrieve(retriever))
    g.add_node("is_relevant", make_is_relevant(llm))
    g.add_node("generate_from_context", make_generate_from_context(llm))
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", make_is_sup(llm))
    g.add_node("revise_answer", make_revise_answer(llm))
    g.add_node("is_use", make_is_use(llm))
    g.add_node("rewrite_question", make_rewrite_question(llm))

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
