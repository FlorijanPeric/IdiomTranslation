import argparse

import uvicorn

from config import ProjectConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local PC RAG server")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable development auto-reload")
    args = parser.parse_args()

    config = ProjectConfig()
    host = args.host or config.server_host
    port = args.port or config.server_port

    uvicorn.run("web_server:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
