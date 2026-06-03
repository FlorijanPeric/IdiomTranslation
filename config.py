from dataclasses import dataclass
import os


@dataclass
class ProjectConfig:
    embedding_api_url: str = os.getenv("EMBEDDING_API_URL", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_api_model: str = os.getenv("EMBEDDING_API_MODEL", "bge-m3")
    embedding_api_timeout: int = int(os.getenv("EMBEDDING_API_TIMEOUT", "120"))
    embedding_batch_size: int = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
    embedding_api_retries: int = max(0, int(os.getenv("EMBEDDING_API_RETRIES", "2")))
    embedding_api_backoff_sec: float = max(0.0, float(os.getenv("EMBEDDING_API_BACKOFF_SEC", "1.5")))

    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "Beees")

    data_root: str = os.getenv("IDIOM_DATA_ROOT", ".")

    chat_api_url: str = os.getenv("CHAT_API_URL", "")
    chat_api_key: str = os.getenv("CHAT_API_KEY", "")
    chat_api_model: str = os.getenv("CHAT_API_MODEL", "")
    chat_api_timeout: int = int(os.getenv("CHAT_API_TIMEOUT", "120"))
    chat_top_k: int = max(1, int(os.getenv("CHAT_TOP_K", "5")))
    chat_history_turns: int = max(0, int(os.getenv("CHAT_HISTORY_TURNS", "4")))

    retrieval_min_score: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.15"))

    gui_host: str = os.getenv("GUI_HOST", "127.0.0.1")
    gui_port: int = int(os.getenv("GUI_PORT", "7860"))

    def validate_embedding_config(self) -> None:
        if not self.embedding_api_url:
            raise RuntimeError("EMBEDDING_API_URL is required.")
        if not self.embedding_api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required.")

    def can_use_chat_api(self) -> bool:
        return bool(self.chat_api_url and self.chat_api_model)

    def validate_chat_config(self) -> None:
        if not self.chat_api_url:
            raise RuntimeError("CHAT_API_URL is required for chat API usage.")
        if not self.chat_api_model:
            raise RuntimeError("CHAT_API_MODEL is required for chat API usage.")
