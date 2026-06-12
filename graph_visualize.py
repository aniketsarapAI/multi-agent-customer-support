import warnings
warnings.filterwarnings("ignore")

from app.graph.builder import build_app


def main():
    app = build_app()
    g = app.get_graph()
    png_data = g.draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("✅ graph.png saved")


if __name__ == "__main__":
    main()
