import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Idiom RAG FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as error:
        raise RuntimeError(
            "Unable to start uvicorn. Install requirements and use Python 3.10-3.12."
        ) from error

    uvicorn.run("web_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
