from typing import Literal

from langchain_core.documents import Document

from app.state import State
from app.models import (
    RetrieveDecision,
    RelevanceDecision,
    IsSUPDecision,
    IsUSEDecision,
    RewriteDecision,
)
from app.prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    isuse_prompt,
    rewrite_for_retrieval_prompt,
)
from app.config import MAX_RETRIES, MAX_REWRITE_TRIES
from app.chat_history import format_chat_history

# These are set at build time to avoid circular imports
_llm = None
_retriever = None


def set_retriever(retriever):
    global _retriever
    _retriever = retriever


def set_llm(llm):
    global _llm
    _llm = llm


# ---------------------------------------------------------------------------
# 1) Decide retrieval
# ---------------------------------------------------------------------------
def decide_retrieval(state: State):
    logs = ["🔍 decide_retrieval: checking if retrieval is needed..."]
    structured = _llm.with_structured_output(RetrieveDecision)
    decision: RetrieveDecision = structured.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    logs.append(f"✅ decide_retrieval: need_retrieval={decision.should_retrieve}")
    return {"need_retrieval": decision.should_retrieve, "logs": logs}


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


# ---------------------------------------------------------------------------
# 2) Direct answer (no retrieval)
# ---------------------------------------------------------------------------
def generate_direct(state: State):
    logs = ["🔍 generate_direct: answering from general knowledge..."]
    chat_history = format_chat_history(state.get("chat_history", []))
    out = _llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"],
            chat_history=chat_history,
        )
    )
    logs.append("✅ generate_direct: done")
    return {
        "answer": out.content,
        "logs": logs,
        "issup": "fully_supported",
        "isuse": "useful",
        "use_reason": "Directly answered from general knowledge or conversational response.",
    }


# ---------------------------------------------------------------------------
# 3) Retrieve
# ---------------------------------------------------------------------------
def retrieve(state: State):
    q = state.get("retrieval_query") or state["question"]
    docs = _retriever.invoke(q)
    return {"docs": docs, "logs": [f"🔍 retrieve: found {len(docs)} document(s)"]}


# ---------------------------------------------------------------------------
# 4) Relevance filter
# ---------------------------------------------------------------------------
def is_relevant(state: State):
    logs = ["🔍 is_relevant: filtering documents by relevance..."]
    structured = _llm.with_structured_output(RelevanceDecision)
    relevant_docs: list[Document] = []
    for i, doc in enumerate(state.get("docs", [])):
        decision: RelevanceDecision = structured.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
        logs.append(f"  doc[{i}]: {'relevant' if decision.is_relevant else 'irrelevant'}")
    logs.append(f"✅ is_relevant: {len(relevant_docs)}/{len(state.get('docs', []))} relevant")
    return {"relevant_docs": relevant_docs, "logs": logs}


def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


# ---------------------------------------------------------------------------
# 5) Generate from context
# ---------------------------------------------------------------------------
def generate_from_context(state: State):
    logs = ["🔍 generate_from_context: writing answer from retrieved docs..."]
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()
    if not context:
        logs.append("⚠️ generate_from_context: no context available")
        return {"answer": "No answer found.", "context": "", "logs": logs}
    out = _llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"], context=context
        )
    )
    logs.append("✅ generate_from_context: done")
    return {"answer": out.content, "context": context, "logs": logs}


def no_answer_found(state: State):
    return {"answer": "No answer found.", "context": "", "logs": ["⚠️ no_answer_found: no relevant documents"]}


# ---------------------------------------------------------------------------
# 6) IsSUP verify + revise loop
# ---------------------------------------------------------------------------
def is_sup(state: State):
    retry = state.get("retries", 0)
    logs = [f"🔍 is_sup (attempt {retry + 1}): verifying answer support..."]
    structured = _llm.with_structured_output(IsSUPDecision)
    decision: IsSUPDecision = structured.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    logs.append(f"✅ is_sup: {decision.issup}")
    return {"issup": decision.issup, "evidence": decision.evidence, "logs": logs}


def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"


def accept_answer(state: State):
    return {"logs": ["✅ accept_answer: answer fully supported"]}


def revise_answer(state: State):
    retry = state.get("retries", 0) + 1
    logs = [f"🔍 revise_answer (attempt {retry}): revising answer..."]
    out = _llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    logs.append(f"✅ revise_answer: done (attempt {retry})")
    return {
        "answer": out.content,
        "retries": retry,
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# 7) IsUSE
# ---------------------------------------------------------------------------
def is_use(state: State):
    logs = ["🔍 is_use: checking answer usefulness..."]
    structured = _llm.with_structured_output(IsUSEDecision)
    decision: IsUSEDecision = structured.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    logs.append(f"✅ is_use: {decision.isuse} — {decision.reason}")
    return {"isuse": decision.isuse, "use_reason": decision.reason, "logs": logs}


def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


# ---------------------------------------------------------------------------
# 8) Rewrite question
# ---------------------------------------------------------------------------
def rewrite_question(state: State):
    rewrite_try = state.get("rewrite_tries", 0) + 1
    logs = [f"🔍 rewrite_question (attempt {rewrite_try}): rewriting for better retrieval..."]
    chat_history = format_chat_history(state.get("chat_history", []))
    structured = _llm.with_structured_output(RewriteDecision)
    decision: RewriteDecision = structured.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            chat_history=chat_history,
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    logs.append(f"✅ rewrite_question: → {decision.retrieval_query}")
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": rewrite_try,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "logs": logs,
    }
