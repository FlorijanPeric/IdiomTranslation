from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from build_index import build_index
from chatbot_gui import _call_chat_api, _fallback_response
from config import ProjectConfig
from retriever import retrieve_candidates


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class TrainRequest(BaseModel):
    recreate_collection: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="Idiom RAG Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        index_file = Path(__file__).resolve().parent / "web" / "index.html"
        return index_file.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        config = ProjectConfig()
        client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)

        collections = client.get_collections().collections
        collection_exists = any(item.name == config.qdrant_collection for item in collections)

        count = 0
        if collection_exists:
            count_result = client.count(collection_name=config.qdrant_collection, exact=False)
            count = int(count_result.count)

        return {
            "ok": True,
            "collection": config.qdrant_collection,
            "collection_exists": collection_exists,
            "points": count,
        }

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> dict:
        config = ProjectConfig()
        history_dicts = [item.model_dump() for item in payload.history]

        try:
            candidates = retrieve_candidates(payload.message, config, top_k=config.chat_top_k)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {error}") from error

        answer = ""
        if config.can_use_chat_api():
            try:
                answer = _call_chat_api(payload.message, history_dicts, candidates, config)
            except Exception:
                answer = _fallback_response(payload.message, candidates)
        else:
            answer = _fallback_response(payload.message, candidates)

        return {"answer": answer, "candidates": candidates}

    @app.post("/api/train")
    def train(payload: TrainRequest) -> dict:
        config = ProjectConfig()
        try:
            return build_index(config, recreate_collection=payload.recreate_collection)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Training failed: {error}") from error

    return app


app = create_app()
