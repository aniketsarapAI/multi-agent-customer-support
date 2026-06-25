import time

from app.models.agent import BaseAgent, AgentResult
from app.models.metadata import RAGMetadata
from app.models.state import RAGAgentState
from app.agents.graph.rag_subgraph import build_rag_subgraph


class RAGAgent(BaseAgent):
    def __init__(self, llm, retriever):
        self._llm = llm
        self._retriever = retriever
        self._subgraph = build_rag_subgraph(llm, retriever)

    def invoke(self, question: str, chat_history: list[dict], request_id: str) -> AgentResult:
        start = time.perf_counter()

        initial_state: RAGAgentState = {
            "request_id": request_id,
            "question": question,
            "answer": "",
            "retrieval_query": "",
            "rewrite_tries": 0,
            "need_retrieval": False,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "issup": "no_support",
            "evidence": [],
            "retries": 0,
            "isuse": "not_useful",
            "use_reason": "",
            "logs": [],
        }

        result = self._subgraph.invoke(initial_state)
        elapsed = int((time.perf_counter() - start) * 1000)

        logs = result.get("logs", [])
        if not isinstance(logs, list):
            logs = [str(logs)]

        metadata = RAGMetadata(
            retrieved_docs=len(result.get("docs", []) or []),
            relevant_docs=len(result.get("relevant_docs", []) or []),
            issup=result.get("issup", ""),
            isuse=result.get("isuse", ""),
            use_reason=result.get("use_reason", ""),
            rewrite_tries=result.get("rewrite_tries", 0),
            retries=result.get("retries", 0),
        )

        return AgentResult(
            agent="rag",
            success=True,
            answer=result.get("answer", "No answer found."),
            confidence=1.0 if result.get("isuse") == "useful" else 0.5,
            metadata=metadata,
            latency_ms=elapsed,
            logs=logs,
        )

    @property
    def graph(self):
        return self._subgraph.get_graph()

    def health(self) -> bool:
        return self._subgraph is not None

    def capabilities(self) -> list[str]:
        return ["rag", "document_retrieval", "self_verification"]
