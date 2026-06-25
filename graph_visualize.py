import warnings
warnings.filterwarnings("ignore")

from app.graph.builder import Application


def main():
    app = Application()

    # Supervisor graph (top-level)
    png = app.supervisor.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png)
    print("✅ graph.png saved (supervisor graph)")

    # RAG subgraph (first agent)
    rag = app.registry.get("rag")
    if hasattr(rag, "_subgraph") and rag._subgraph is not None:
        png = rag._subgraph.get_graph().draw_mermaid_png()
        with open("graph_rag.png", "wb") as f:
            f.write(png)
        print("✅ graph_rag.png saved (RAG agent subgraph)")

    # SQL subgraph
    sql = app.registry.get("sql")
    if hasattr(sql, "_subgraph") and sql._subgraph is not None:
        png = sql._subgraph.get_graph().draw_mermaid_png()
        with open("graph_sql.png", "wb") as f:
            f.write(png)
        print("✅ graph_sql.png saved (SQL agent subgraph)")


if __name__ == "__main__":
    main()
