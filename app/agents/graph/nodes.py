from typing import Literal

from langchain_core.documents import Document

from app.models.state import RAGAgentState
from app.models.metadata import (
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


def make_decide_retrieval(grader_llm):
    def decide_retrieval(state: RAGAgentState):
        logs = ["🔍 decide_retrieval: checking if retrieval is needed..."]
        chat_history = format_chat_history(state.get("chat_history", []))
        structured = grader_llm.with_structured_output(
            RetrieveDecision, default=lambda: RetrieveDecision(should_retrieve=True)
        )
        decision: RetrieveDecision = structured.invoke(
            decide_retrieval_prompt.format_messages(
                question=state["question"],
                chat_history=chat_history,
            )
        )
        logs.append(f"✅ decide_retrieval: need_retrieval={decision.should_retrieve}")
        return {"need_retrieval": decision.should_retrieve, "logs": logs}
    return decide_retrieval


def route_after_decide(state: RAGAgentState) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def make_generate_direct(llm):
    def generate_direct(state: RAGAgentState):
        logs = ["🔍 generate_direct: answering from general knowledge..."]
        chat_history = format_chat_history(state.get("chat_history", []))
        out = llm.invoke(
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
    return generate_direct


def make_retrieve(retriever):
    def retrieve(state: RAGAgentState):
        q = state.get("retrieval_query") or state["question"]
        docs = retriever.invoke(q)
        return {"docs": docs, "logs": [f"🔍 retrieve: found {len(docs)} document(s)"]}
    return retrieve


def make_is_relevant(llm):
    def is_relevant(state: RAGAgentState):
        logs = ["🔍 is_relevant: filtering documents by relevance..."]
        structured = llm.with_structured_output(
            RelevanceDecision, default=lambda: RelevanceDecision(is_relevant=True)
        )
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
    return is_relevant


def route_after_relevance(state: RAGAgentState) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


def make_generate_from_context(llm):
    def generate_from_context(state: RAGAgentState):
        logs = ["🔍 generate_from_context: writing answer from retrieved docs..."]
        context = "\n\n---\n\n".join(
            [d.page_content for d in state.get("relevant_docs", [])]
        ).strip()
        if not context:
            logs.append("⚠️ generate_from_context: no context available")
            return {"answer": "No answer found.", "context": "", "logs": logs}
        out = llm.invoke(
            rag_generation_prompt.format_messages(
                question=state["question"], context=context
            )
        )
        logs.append("✅ generate_from_context: done")
        return {"answer": out.content, "context": context, "logs": logs}
    return generate_from_context


def no_answer_found(state: RAGAgentState):
    return {"answer": "No answer found.", "context": "", "logs": ["⚠️ no_answer_found: no relevant documents"]}


def make_is_sup(llm):
    def is_sup(state: RAGAgentState):
        retry = state.get("retries", 0)
        logs = [f"🔍 is_sup (attempt {retry + 1}): verifying answer support..."]
        structured = llm.with_structured_output(
            IsSUPDecision, default=lambda: IsSUPDecision(issup="partially_supported", evidence=[])
        )
        decision: IsSUPDecision = structured.invoke(
            issup_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        logs.append(f"✅ is_sup: {decision.issup}")
        return {"issup": decision.issup, "evidence": decision.evidence, "logs": logs}
    return is_sup


def route_after_issup(state: RAGAgentState) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup", "").lower() == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"


def accept_answer(state: RAGAgentState):
    return {"logs": ["✅ accept_answer: answer fully supported"]}


def make_revise_answer(llm):
    def revise_answer(state: RAGAgentState):
        retry = state.get("retries", 0) + 1
        logs = [f"🔍 revise_answer (attempt {retry}): revising answer..."]
        out = llm.invoke(
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
    return revise_answer


def make_is_use(llm):
    def is_use(state: RAGAgentState):
        logs = ["🔍 is_use: checking answer usefulness..."]
        structured = llm.with_structured_output(
            IsUSEDecision, default=lambda: IsUSEDecision(isuse="not_useful", reason="Usefulness grader failed.")
        )
        decision: IsUSEDecision = structured.invoke(
            isuse_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
            )
        )
        logs.append(f"✅ is_use: {decision.isuse} — {decision.reason}")
        return {"isuse": decision.isuse, "use_reason": decision.reason, "logs": logs}
    return is_use


def route_after_isuse(state: RAGAgentState) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse", "").lower() == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


def make_rewrite_question(llm):
    def rewrite_question(state: RAGAgentState):
        rewrite_try = state.get("rewrite_tries", 0) + 1
        logs = [f"🔍 rewrite_question (attempt {rewrite_try}): rewriting for better retrieval..."]
        chat_history = format_chat_history(state.get("chat_history", []))
        structured = llm.with_structured_output(
            RewriteDecision, default=lambda: RewriteDecision(retrieval_query="")
        )
        decision: RewriteDecision = structured.invoke(
            rewrite_for_retrieval_prompt.format_messages(
                question=state["question"],
                chat_history=chat_history,
                retrieval_query=state.get("retrieval_query", ""),
                answer=state.get("answer", ""),
            )
        )
        new_query = decision.retrieval_query or state["question"]
        logs.append(f"✅ rewrite_question: → {new_query}")
        return {
            "retrieval_query": new_query,
            "rewrite_tries": rewrite_try,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "logs": logs,
        }
    return rewrite_question
