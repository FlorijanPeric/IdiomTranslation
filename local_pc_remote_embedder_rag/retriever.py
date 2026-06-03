from config import ProjectConfig
from embedder_api import embed_texts
from qdrant_api import QdrantHttpClient


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _lexical_score(query: str, idiom: str, meaning: str, translation: str) -> float:
    query_tokens = set(_tokenize(query))
    idiom_tokens = set(_tokenize(idiom))
    meaning_tokens = set(_tokenize(meaning))
    translation_tokens = set(_tokenize(translation))

    idiom_sim = _jaccard_similarity(query_tokens, idiom_tokens)
    meaning_sim = _jaccard_similarity(query_tokens, meaning_tokens)
    translation_sim = _jaccard_similarity(query_tokens, translation_tokens)

    return max(idiom_sim, 0.7 * meaning_sim, 0.6 * translation_sim)


def _rule_boost(query_norm: str, idiom_norm: str, config: ProjectConfig) -> float:
    if not query_norm or not idiom_norm:
        return 0.0
    if idiom_norm == query_norm:
        return config.retrieval_exact_match_boost
    if idiom_norm.startswith(query_norm):
        return config.retrieval_prefix_match_boost
    if query_norm in idiom_norm:
        return config.retrieval_contains_match_boost
    return 0.0


def retrieve_candidates(
    query: str,
    config: ProjectConfig,
    top_k: int | None = None,
    direction: str = "en_to_sl",
) -> list[dict]:
    k = top_k or config.chat_top_k
    query_vector = embed_texts([query], config)[0]
    query_norm = _normalize(query)
    collection_name = config.collection_for_direction(direction)

    client = QdrantHttpClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
    candidate_limit = max(k * config.retrieval_candidate_multiplier, config.retrieval_candidate_min_limit)
    hits = client.search(collection_name=collection_name, query_vector=query_vector, limit=candidate_limit)

    rows: list[dict] = []
    for hit in hits:
        payload = hit.get("payload") or {}
        vector_score = float(hit.get("score", 0.0))
        idiom = str(payload.get("idiom", ""))
        meaning = str(payload.get("meaning", ""))
        translation = str(payload.get("target_translation", ""))
        idiom_norm = _normalize(payload.get("idiom_norm") or idiom)

        lexical = _lexical_score(query, idiom, meaning, translation)
        boost = _rule_boost(query_norm, idiom_norm, config)

        adjusted_score = (
            config.retrieval_hybrid_vector_weight * vector_score
            + config.retrieval_hybrid_lexical_weight * lexical
            + boost
        )
        if adjusted_score < config.retrieval_min_score:
            continue
        rows.append(
            {
                "score": adjusted_score,
                "raw_score": vector_score,
                "lexical_score": lexical,
                "rule_boost": boost,
                "idiom": idiom,
                "meaning": meaning,
                "translation": translation,
                "source_file": str(payload.get("source_file", "")),
            }
        )

    rows = sorted(rows, key=lambda item: item["score"], reverse=True)
    return rows[:k]
