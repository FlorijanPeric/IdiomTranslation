import json
import time
import math
import socket
from urllib import request
from urllib.error import HTTPError, URLError

from config import ProjectConfig


def _parse_embedding_response(payload: str, expected_count: int) -> list[list[float]]:
    parsed = json.loads(payload)

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = parsed.get("data") or parsed.get("embeddings") or parsed.get("results") or parsed.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Embedding API response format is invalid.")
    else:
        raise RuntimeError("Embedding API response format is invalid.")

    vectors: list[list[float]] = []
    for item in items:
        if isinstance(item, list):
            vectors.append([float(value) for value in item])
            continue
        if isinstance(item, dict):
            vector = item.get("embedding") or item.get("vector")
            if isinstance(vector, list):
                vectors.append([float(value) for value in vector])
                continue
        raise RuntimeError("Embedding API item must be a vector array or object with embedding/vector.")

    if len(vectors) != expected_count:
        raise RuntimeError(f"Embedding count mismatch: expected {expected_count}, got {len(vectors)}")

    for vector in vectors:
        if not vector:
            raise RuntimeError("Embedding API returned empty vector.")
        if any((not math.isfinite(value)) for value in vector):
            raise RuntimeError("Embedding API returned non-finite vector values.")

    return vectors


def embed_texts(texts: list[str], config: ProjectConfig) -> list[list[float]]:
    if not texts:
        return []
    config.validate_embedding_config()

    payload = {
        "model": config.embedding_api_model,
        "input": texts,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.embedding_api_key}",
    }

    req = request.Request(
        url=config.embedding_api_url,
        data=body,
        headers=headers,
        method="POST",
    )

    last_error: Exception | None = None
    max_attempts = config.embedding_api_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            with request.urlopen(req, timeout=config.embedding_api_timeout) as response:
                raw = response.read().decode("utf-8")
            return _parse_embedding_response(raw, expected_count=len(texts))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"Embedding API error ({error.code}): {detail}")
        except URLError as error:
            last_error = RuntimeError(f"Embedding API connection failed: {error}")
        except (TimeoutError, socket.timeout) as error:
            last_error = RuntimeError("Embedding API request timed out.")
        except Exception as error:
            last_error = RuntimeError(f"Embedding API unexpected failure: {error}")

        if attempt < max_attempts:
            sleep_for = config.embedding_api_backoff_sec * attempt
            print(f"Embedding batch failed (attempt {attempt}/{max_attempts}), retrying in {sleep_for:.1f}s...")
            time.sleep(sleep_for)

    raise RuntimeError(str(last_error))


def embed_in_batches(texts: list[str], config: ProjectConfig) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.embedding_batch_size):
        batch = texts[start : start + config.embedding_batch_size]
        vectors.extend(embed_texts(batch, config))
    return vectors
