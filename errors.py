"""Domain exceptions for the MMA RAG system.

Every error meant for the end user derives from ``RAGError`` so the CLI can
show a friendly message without a stack trace (unless ``--debug`` is used).
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for user-facing errors."""


class ConfigError(RAGError):
    """Missing or invalid configuration (environment variables, ranges)."""


class InvalidInputError(RAGError):
    """The user's question failed validation (empty, too long, ...)."""


class IndexNotReadyError(RAGError):
    """The Pinecone index does not exist or contains no vectors."""


class UpstreamError(RAGError):
    """An external API (Gemini or Pinecone) failed."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
