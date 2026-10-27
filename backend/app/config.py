"""Application configuration, loaded from environment variables / a `.env` file.

Every setting can be overridden with an environment variable prefixed `MEDQUERY_`,
e.g. `MEDQUERY_QUERY_MODEL_NAME=dmis-lab/biobert-base-cased-v1.1`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDQUERY_",
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
    )

    app_name: str = "MedQuery"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # --- Request limits -------------------------------------------------------
    max_question_chars: int = 500
    min_question_chars: int = 3
    max_request_bytes: int = 8_192

    # --- BERT query understanding ---------------------------------------------
    # Any BERT-family encoder on the Hugging Face Hub (or a local directory).
    query_model_name: str = "bert-base-uncased"
    query_max_tokens: int = 128
    # Load models at startup (in a background thread) instead of on first use.
    preload_models: bool = False
    # If true, never contact the Hugging Face Hub; use only locally cached weights.
    hf_offline: bool = False

    # --- Knowledge retrieval (YOUR knowledge base plugs in here) -------------
    # Import path "package.module:ClassName" of a class implementing
    # retrieval.base.Retriever. Empty -> NotConfiguredRetriever.
    kb_adapter: str = ""
    # Free-form connection string handed to your adapter (file path, DB URL, ...).
    kb_connection: str = ""
    kb_top_k: int = 5
    kb_timeout_seconds: float = 10.0
    # Passages scoring below this are discarded as not relevant enough.
    kb_min_relevance: float = 0.0

    # --- Answer processing ----------------------------------------------------
    # "extractive": BERT reading-comprehension model selects an answer span.
    # "evidence_only": no answer synthesis; the best passages are returned.
    answer_mode: Literal["extractive", "evidence_only"] = "extractive"
    qa_model_name: str = "deepset/bert-base-cased-squad2"
    # Extractive answers with a lower confidence are flagged as low confidence.
    qa_min_confidence: float = 0.15
    qa_max_answer_tokens: int = 64

    # --- Persistence ----------------------------------------------------------
    database_path: str = str(BACKEND_DIR / "data" / "medquery.sqlite3")
    history_limit: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
