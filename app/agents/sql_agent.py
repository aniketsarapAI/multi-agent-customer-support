import time

from app.models.agent import BaseAgent, AgentResult
from app.models.metadata import SQLMetadata
from app.models.state import SQLAgentState
from app.agents.graph.sql_nodes import build_sql_subgraph


class SQLAgent(BaseAgent):
    def __init__(self, llm):
        self._llm = llm
        self._subgraph = build_sql_subgraph(llm)

    def invoke(self, question: str, chat_history: list[dict], request_id: str) -> AgentResult:
        start = time.perf_counter()

        initial_state: SQLAgentState = {
            "request_id": request_id,
            "question": question,
            "answer": "",
            "sql_query": "",
            "sql_result": "",
            "db_error": "",
            "visualization_spec": None,
            "logs": [],
        }

        result = self._subgraph.invoke(initial_state)
        elapsed = int((time.perf_counter() - start) * 1000)

        logs = result.get("logs", [])
        if not isinstance(logs, list):
            logs = [str(logs)]

        metadata = SQLMetadata(
            sql_query=result.get("sql_query", ""),
            sql_result=result.get("sql_result", ""),
            db_error=result.get("db_error", ""),
            visualization_spec=result.get("visualization_spec"),
        )

        success = not bool(result.get("db_error"))
        answer = result.get("answer") or result.get("db_answer") or "No answer found."

        return AgentResult(
            agent="sql",
            success=success,
            answer=answer,
            confidence=1.0 if success else 0.0,
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
        return ["sql", "database_query", "visualization"]
