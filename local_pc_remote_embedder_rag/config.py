from dataclasses import dataclass
import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) >= 2 and ((trimmed.startswith('"') and trimmed.endswith('"')) or (trimmed.startswith("'") and trimmed.endswith("'"))):
        return trimmed[1:-1]
    return trimmed


def _load_env_file(file_path: Path, override: bool) -> None:
    if not file_path.exists() or not file_path.is_file():
        return

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        parsed_value = _strip_quotes(value)
        if override or key not in os.environ:
            os.environ[key] = parsed_value


def _first_non_empty_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


_BASE_DIR = Path(__file__).resolve().parent
_PARENT_ENV_FILE = _BASE_DIR.parent / ".env"
_LOCAL_ENV_FILE = _BASE_DIR / ".env"

_load_env_file(_PARENT_ENV_FILE, override=False)
_load_env_file(_LOCAL_ENV_FILE, override=True)


@dataclass
class ProjectConfig:
    embedding_api_url: str = os.getenv("EMBEDDING_API_URL", "")
    embedding_api_key: str = _first_non_empty_env(
        "EMBEDDING_API_KEY",
        "EMBEDDING_BEARER_TOKEN",
        "BEARER_TOKEN",
    )
    embedding_api_model: str = os.getenv("EMBEDDING_API_MODEL", "bge-m3")
    embedding_api_timeout: int = int(os.getenv("EMBEDDING_API_TIMEOUT", "120"))
    embedding_batch_size: int = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
    embedding_api_retries: int = max(0, int(os.getenv("EMBEDDING_API_RETRIES", "2")))
    embedding_api_backoff_sec: float = max(0.0, float(os.getenv("EMBEDDING_API_BACKOFF_SEC", "1.5")))

    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "idioms_local_pc")
    qdrant_collection_reverse: str = os.getenv("QDRANT_COLLECTION_REVERSE", "JezikovneTehnologijeReverse")

    data_roots_raw: str = os.getenv("IDIOM_DATA_ROOTS", os.getenv("IDIOM_DATA_ROOT", "./data,.."))

    retrieval_min_score: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.15"))
    retrieval_hybrid_vector_weight: float = float(os.getenv("RETRIEVAL_HYBRID_VECTOR_WEIGHT", "0.75"))
    retrieval_hybrid_lexical_weight: float = float(os.getenv("RETRIEVAL_HYBRID_LEXICAL_WEIGHT", "0.25"))
    retrieval_exact_match_boost: float = float(os.getenv("RETRIEVAL_EXACT_MATCH_BOOST", "0.12"))
    retrieval_prefix_match_boost: float = float(os.getenv("RETRIEVAL_PREFIX_MATCH_BOOST", "0.08"))
    retrieval_contains_match_boost: float = float(os.getenv("RETRIEVAL_CONTAINS_MATCH_BOOST", "0.05"))
    retrieval_candidate_multiplier: int = max(2, int(os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "4")))
    retrieval_candidate_min_limit: int = max(10, int(os.getenv("RETRIEVAL_CANDIDATE_MIN_LIMIT", "20")))
    chat_top_k: int = max(1, int(os.getenv("CHAT_TOP_K", "5")))

    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))
    
    translation_direction: str = os.getenv("TRANSLATION_DIRECTION", "en_to_sl")

    def validate_embedding_config(self) -> None:
        if not self.embedding_api_url:
            raise RuntimeError("EMBEDDING_API_URL is required.")
        if not self.embedding_api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required.")

    def data_roots(self) -> list[str]:
        values = [item.strip() for item in self.data_roots_raw.split(",") if item.strip()]
        if not values:
            return ["./data", ".."]
        return values

    def collection_for_direction(self, direction: str) -> str:
        if direction == "sl_to_en":
            return self.qdrant_collection_reverse
        return self.qdrant_collection
