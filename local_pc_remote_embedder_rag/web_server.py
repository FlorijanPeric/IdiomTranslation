from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from build_index import build_index
from config import ProjectConfig
from qdrant_api import QdrantHttpClient
from retriever import retrieve_candidates


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    llm_output: str | None = None


class TranslateRequest(BaseModel):
    idiom: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    llm_output: str | None = None
    direction: str = Field(default="en_to_sl", pattern="^(en_to_sl|sl_to_en)$")


class TrainRequest(BaseModel):
    recreate_collection: bool = False
    direction: str = Field(default="en_to_sl", pattern="^(en_to_sl|sl_to_en)$")


class TestEmbedderRequest(BaseModel):
    sample_text: str = Field(default="A bad hair day")


def _clean_translation(text: str) -> str:
    value = str(text).strip().strip('"').strip("'")
    if not value:
        return ""

    for separator in ["/", ";", "|"]:
        if separator in value:
            first = value.split(separator, 1)[0].strip()
            if first:
                value = first
    return value.strip()


def _best_translation(candidates: list[dict]) -> str:
    for candidate in candidates:
        cleaned = _clean_translation(candidate.get("translation", ""))
        if cleaned:
            return cleaned
    return ""


def _postprocess_llm_output(llm_output: str | None, candidates: list[dict]) -> str:
    best = _best_translation(candidates)
    if not llm_output:
        return best

    content = llm_output.lower()
    for candidate in candidates:
        raw = str(candidate.get("translation", "")).strip()
        cleaned = _clean_translation(raw)
        if not cleaned:
            continue
        if cleaned.lower() in content:
            return cleaned
        if raw and raw.lower() in content:
            return cleaned

    return best


def _fallback_response(candidates: list[dict]) -> str:
    if not candidates:
        return "No confident idiom match found in the current index."

    best = candidates[0]
    lines = [
        f"Best match: {best['idiom']}",
        f"Translation: {best['translation']}",
        f"Meaning: {best['meaning']}",
    ]
    if len(candidates) > 1:
        lines.append("\nOther close matches:")
        for candidate in candidates[1:3]:
            lines.append(f"- {candidate['idiom']} -> {candidate['translation']}")
    return "\n".join(lines)


def create_app() -> FastAPI:
    app = FastAPI(title="Local PC Idiom RAG", version="1.0.0")

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
        client = QdrantHttpClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
        collections = client.list_collections()
        en_collection = config.collection_for_direction("en_to_sl")
        sl_collection = config.collection_for_direction("sl_to_en")

        en_exists = en_collection in collections
        sl_exists = sl_collection in collections

        en_count = client.count_points(collection_name=en_collection) if en_exists else 0
        sl_count = client.count_points(collection_name=sl_collection) if sl_exists else 0

        return {
            "ok": True,
            "collection": en_collection,
            "collection_exists": en_exists,
            "points": en_count,
            "collections": {
                "en_to_sl": {
                    "name": en_collection,
                    "exists": en_exists,
                    "points": en_count,
                },
                "sl_to_en": {
                    "name": sl_collection,
                    "exists": sl_exists,
                    "points": sl_count,
                },
            },
            "embedder_url": config.embedding_api_url,
            "embedder_model": config.embedding_api_model,
            "embedder_token_set": bool(config.embedding_api_key),
            "data_roots": config.data_roots(),
        }

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> dict:
        config = ProjectConfig()
        try:
            candidates = retrieve_candidates(payload.message, config, top_k=config.chat_top_k)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {error}") from error

        answer = _fallback_response(candidates)
        best_translation = _best_translation(candidates)
        postprocessed_translation = _postprocess_llm_output(payload.llm_output, candidates)

        return {
            "answer": answer,
            "candidates": candidates,
            "best_translation": best_translation,
            "postprocessed_translation": postprocessed_translation,
        }

    @app.post("/api/translate")
    def translate(payload: TranslateRequest) -> dict:
        config = ProjectConfig()
        try:
            candidates = retrieve_candidates(payload.idiom, config, top_k=payload.top_k, direction=payload.direction)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Translation retrieval failed: {error}") from error

        best_translation = _best_translation(candidates)
        postprocessed_translation = _postprocess_llm_output(payload.llm_output, candidates)

        return {
            "query": payload.idiom,
            "direction": payload.direction,
            "best_translation": best_translation,
            "postprocessed_translation": postprocessed_translation,
            "answer": _fallback_response(candidates),
            "candidates": candidates,
        }

    @app.post("/api/train")
    def train(payload: TrainRequest) -> dict:
        config = ProjectConfig()
        try:
            return build_index(config, recreate_collection=payload.recreate_collection, direction=payload.direction)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Training failed: {error}") from error

    @app.post("/api/test-embedder")
    def test_embedder(payload: TestEmbedderRequest) -> dict:
        config = ProjectConfig()
        try:
            from embedder_api import embed_texts

            vectors = embed_texts([payload.sample_text], config)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Embedder test failed: {error}") from error

        vector = vectors[0] if vectors else []
        return {
            "ok": True,
            "sample_text": payload.sample_text,
            "vector_dimension": len(vector),
            "embedder_url": config.embedding_api_url,
            "embedder_model": config.embedding_api_model,
            "token_set": bool(config.embedding_api_key),
        }

    return app


app = create_app()
