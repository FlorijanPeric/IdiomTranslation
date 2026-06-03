from qdrant_client import QdrantClient

from config import ProjectConfig
from embedder_api import embed_texts


def retrieve_candidates(query: str, config: ProjectConfig, top_k: int | None = None) -> list[dict]:
    k = top_k or config.chat_top_k
    query_vector = embed_texts([query], config)[0]

    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
    hits = client.search(
        collection_name=config.qdrant_collection,
        query_vector=query_vector,
        limit=k,
        with_payload=True,
    )

    rows = []
    for hit in hits:
        payload = hit.payload or {}
        score = float(hit.score)
        if score < config.retrieval_min_score:
            continue
        rows.append(
            {
                "score": score,
                "idiom": str(payload.get("idiom", "")),
                "meaning": str(payload.get("meaning", "")),
                "translation": str(payload.get("target_translation", "")),
                "source_file": str(payload.get("source_file", "")),
            }
        )
    return rows
