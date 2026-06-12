import argparse
import sys

from app.graph.builder import build_app
from app.state import State
from app.stream_events import TokenCollector


def invoke_graph(question: str, chat_history: list[dict] | None = None,
                 rag_docs_used: list[str] | None = None,
                 sql_queries_executed: list[str] | None = None) -> dict:
    app = build_app()
    initial_state: State = {
        "question": question,
        "original_question": question,
        "query_type": "document",
        "chat_history": chat_history or [],

        "sub_questions": [],
        "sub_results": [],
        "sub_question": None,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "need_retrieval": False,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
        "logs": [],
        "sql_query": "",
        "sql_result": "",
        "db_answer": "",
        "db_error": "",
        "visualization_spec": None,
        "escalated": False,
        "escalation_reason": "",
        "handoff_summary": "",
        "rag_docs_used": rag_docs_used or [],
        "sql_queries_executed": sql_queries_executed or [],
    }
    final_state = None
    collected_logs = []
    collector = TokenCollector()
    last_answer = ""
    printed_answer_len = 0

    for output in app.stream(
        initial_state,
        stream_mode="values",
        config={"callbacks": [collector]},
    ):
        final_state = output
        new_logs = output.get("logs", [])
        if isinstance(new_logs, list):
            for log in new_logs:
                if log not in collected_logs:
                    collected_logs.append(log)
                    print(f"  {log}")

        # Drain tokens and print progressively
        answer_chars = []
        while not collector._queue.empty():
            answer_chars.append(collector._queue.get_nowait())
        if answer_chars:
            last_answer += "".join(answer_chars)
            # Print new chars inline
            sys.stdout.write("".join(answer_chars))
            sys.stdout.flush()

    collector.mark_done()
    # Drain any remaining tokens
    remaining = []
    while not collector._queue.empty():
        remaining.append(collector._queue.get_nowait())
    if remaining:
        sys.stdout.write("".join(remaining))
        sys.stdout.flush()

    return final_state or {}


def format_output(result: dict, question: str):
    query_type = result.get("query_type", "document")
    lines = []
    lines.append("\n===== EXECUTION RESULT =====\n")
    lines.append(f"Question: {question}")
    lines.append(f"Query type: {query_type}")

    if query_type == "hybrid":
        lines.append(f"\nSub-questions: {result.get('sub_questions', [])}")
        lines.append(f"\nSub-results: {dict(result.get('sub_results', []))}")
        lines.append(f"\nAnswer:\n{result.get('db_answer', '') or result.get('answer', '')}")
    elif query_type == "database":
        lines.append(f"\nSQL Query: {result.get('sql_query', '')}")
        if result.get("db_error"):
            lines.append(f"Error: {result['db_error']}")
        else:
            lines.append(f"\nResult Preview: {(result.get('sql_result', '')[:300] + '...') if len(result.get('sql_result', '')) > 300 else result.get('sql_result', '')}")
        lines.append(f"\nAnswer:\n{result.get('db_answer', '') or result.get('answer', '')}")
    else:
        lines.append(f"Need Retrieval: {result.get('need_retrieval')}")
        lines.append(f"Rewrite tries (retrieval): {result.get('rewrite_tries', 0)}")
        lines.append(f"Support revise tries: {result.get('retries', 0)}")

        lines.append("\nRetrieval:")
        lines.append(f"  Total retrieved docs: {len(result.get('docs', []) or [])}")
        lines.append(f"  Relevant docs: {len(result.get('relevant_docs', []) or [])}")

        relevant_docs = result.get("relevant_docs", []) or []
        if relevant_docs:
            lines.append("\nRelevant docs (source/page):")
            for i, d in enumerate(relevant_docs, 1):
                meta = d.metadata or {}
                src = meta.get("source", "unknown")
                page = meta.get("page", None)
                extra = f", title={meta.get('title', '')}" if meta.get("title") else ""
                if page is not None:
                    lines.append(f"  {i}. source={src}, page={page}{extra}")
                else:
                    lines.append(f"  {i}. source={src}{extra}")

        lines.append("\nVerification (IsSUP):")
        lines.append(f"  issup: {result.get('issup')}")
        evidence = result.get("evidence", []) or []
        if evidence:
            lines.append("  evidence:")
            for e in evidence:
                lines.append(f"   - {e}")
        else:
            lines.append("  evidence: (none)")

        lines.append("\nUsefulness (IsUSE):")
        lines.append(f"  isuse: {result.get('isuse')}")
        lines.append(f"  reason: {result.get('use_reason', '')}")

        lines.append(f"\nFinal Answer:\n{result.get('answer')}")

    if result.get("escalated"):
        lines.append(f"\n⚠️  ESCALATION TRIGGERED — {result['escalation_reason']}")
        if result.get("handoff_summary"):
            lines.append(f"  Handoff:\n{result['handoff_summary']}")

    lines.append("\n===============================\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Self-RAG MCP — query company documents & e-commerce database")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    args = parser.parse_args()

    if args.question:
        result = invoke_graph(args.question)
        print(format_output(result, args.question))
    elif args.interactive:
        print("Self-RAG MCP Interactive Mode (type 'quit' to exit)\n")
        chat_history: list[dict] = []
        rag_docs_used: list[str] = []
        sql_queries_executed: list[str] = []
        while True:
            try:
                q = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            result = invoke_graph(q, chat_history, rag_docs_used, sql_queries_executed)
            answer = result.get("db_answer") or result.get("answer") or "No answer found."
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": answer})
            # Accumulate retrieval/sql metadata across turns
            new_docs = [
                d.metadata.get("title") or d.metadata.get("source", "")
                for d in result.get("relevant_docs", []) or []
                if d is not None
            ]
            new_docs = [t for t in new_docs if t]
            sql_raw = result.get("sql_query", "")
            new_sql = [sql_raw.replace("\n", " ")[:150]] if sql_raw else []
            rag_docs_used = list(dict.fromkeys(rag_docs_used + new_docs))[-5:]
            sql_queries_executed = (sql_queries_executed + new_sql)[-5:]
            print(format_output(result, q))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
