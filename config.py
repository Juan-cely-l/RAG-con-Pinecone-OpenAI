"""Centralized configuration, read from environment variables (and `.env`) with defaults.

All tunable RAG parameters live here so they can be justified, changed and
experimented with without touching the pipeline code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any

from dotenv import load_dotenv

from errors import ConfigError

REQUIRED_ENV_VARS = ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "GOOGLE_API_KEY")


@dataclass(frozen=True)
class Settings:
    """All runtime settings for ingestion, retrieval and generation."""

    # Credentials / infrastructure
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    google_api_key: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    namespace: str = ""

    # Models
    chat_model: str = "gemini-3.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.2

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 4
    score_threshold: float = 0.5

    # Limits
    max_question_chars: int = 500
    max_context_chars: int = 6000

    # Resilience
    max_retries: int = 4
    retry_base_delay: float = 2.0
    request_timeout: float = 60.0

    # Backwards-compatible aliases used by the original code.
    @property
    def gemini_chat_model(self) -> str:
        return self.chat_model

    @property
    def gemini_embedding_model(self) -> str:
        return self.embedding_model


# field name -> environment variable name
_ENV_MAP = {
    "pinecone_api_key": "PINECONE_API_KEY",
    "pinecone_index_name": "PINECONE_INDEX_NAME",
    "google_api_key": "GOOGLE_API_KEY",
    "pinecone_cloud": "PINECONE_CLOUD",
    "pinecone_region": "PINECONE_REGION",
    "namespace": "PINECONE_NAMESPACE",
    "chat_model": "GEMINI_CHAT_MODEL",
    "embedding_model": "GEMINI_EMBEDDING_MODEL",
    "temperature": "RAG_TEMPERATURE",
    "chunk_size": "RAG_CHUNK_SIZE",
    "chunk_overlap": "RAG_CHUNK_OVERLAP",
    "top_k": "RAG_TOP_K",
    "score_threshold": "RAG_SCORE_THRESHOLD",
    "max_question_chars": "RAG_MAX_QUESTION_CHARS",
    "max_context_chars": "RAG_MAX_CONTEXT_CHARS",
    "max_retries": "RAG_MAX_RETRIES",
    "retry_base_delay": "RAG_RETRY_BASE_DELAY",
    "request_timeout": "RAG_REQUEST_TIMEOUT",
}


def _coerce(name: str, raw: str, target_type: Any) -> Any:
    try:
        if target_type in (int, "int"):
            return int(raw)
        if target_type in (float, "float"):
            return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid value for {_ENV_MAP[name]}: {raw!r} (expected {target_type})."
        ) from exc
    return raw


def _validate(settings: Settings, require_keys: bool) -> None:
    if require_keys:
        missing = [
            env for field_name, env in _ENV_MAP.items()
            if env in REQUIRED_ENV_VARS and not getattr(settings, field_name)
        ]
        if missing:
            raise ConfigError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in the values."
            )
    if settings.chunk_size <= 0:
        raise ConfigError("RAG_CHUNK_SIZE must be > 0.")
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ConfigError("RAG_CHUNK_OVERLAP must be >= 0 and smaller than RAG_CHUNK_SIZE.")
    if settings.top_k < 1:
        raise ConfigError("RAG_TOP_K must be at least 1.")
    if not 0.0 <= settings.score_threshold <= 1.0:
        raise ConfigError("RAG_SCORE_THRESHOLD must be between 0 and 1.")
    if settings.max_question_chars < 1 or settings.max_context_chars < 1:
        raise ConfigError("RAG_MAX_QUESTION_CHARS and RAG_MAX_CONTEXT_CHARS must be positive.")
    if settings.request_timeout <= 0:
        raise ConfigError("RAG_REQUEST_TIMEOUT must be > 0 seconds.")
    if not 0.0 <= settings.temperature <= 2.0:
        raise ConfigError("RAG_TEMPERATURE must be between 0 and 2.")
    if settings.max_retries < 0 or settings.retry_base_delay < 0:
        raise ConfigError("RAG_MAX_RETRIES and RAG_RETRY_BASE_DELAY must be >= 0.")


def load_settings(require_keys: bool = True, use_dotenv: bool = True, **overrides: Any) -> Settings:
    """Build ``Settings`` from env vars (+ `.env`), applying explicit overrides last.

    Raises ``ConfigError`` with the names of missing variables or invalid values.
    """
    if use_dotenv:
        load_dotenv()

    values: dict[str, Any] = {}
    for f in fields(Settings):
        raw = os.getenv(_ENV_MAP[f.name], "").strip()
        if raw:
            values[f.name] = _coerce(f.name, raw, f.type)

    settings = Settings(**values)
    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    if clean_overrides:
        settings = replace(settings, **clean_overrides)

    _validate(settings, require_keys)
    return settings
