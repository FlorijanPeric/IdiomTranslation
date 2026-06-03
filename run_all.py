import argparse

from qdrant_client import QdrantClient

from build_index import build_index
from chatbot_gui import main as chatbot_main
from config import ProjectConfig


def _collection_has_points(config: ProjectConfig) -> bool:
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
    collections = client.get_collections().collections
    if not any(collection.name == config.qdrant_collection for collection in collections):
        return False

    count_result = client.count(collection_name=config.qdrant_collection, exact=False)
    return int(count_result.count) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run idiom project pipeline and GUI chatbot")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building step")
    parser.add_argument("--force-reindex", action="store_true", help="Force rebuilding index")
    parser.add_argument(
        "--ui",
        choices=["gradio", "server"],
        default="server",
        help="UI mode to start after indexing",
    )
    parser.add_argument("--server-host", default="0.0.0.0", help="FastAPI server host")
    parser.add_argument("--server-port", type=int, default=8000, help="FastAPI server port")
    args = parser.parse_args()

    config = ProjectConfig()

    should_index = not args.skip_index
    if should_index and not args.force_reindex and _collection_has_points(config):
        should_index = False

    if should_index:
        print("Building Qdrant index...")
        result = build_index(config, recreate_collection=args.force_reindex)
        print(result)
    else:
        print("Skipping index build.")

    if args.ui == "gradio":
        print("Starting Gradio chatbot UI...")
        chatbot_main()
        return

    try:
        import uvicorn
        from web_server import app as web_app
    except Exception as error:
        raise RuntimeError(
            "FastAPI server dependencies are unavailable. "
            "Install requirements and use Python 3.10-3.12 for server mode."
        ) from error

    print(f"Starting FastAPI server UI at http://{args.server_host}:{args.server_port} ...")
    uvicorn.run(web_app, host=args.server_host, port=args.server_port)


if __name__ == "__main__":
    main()
