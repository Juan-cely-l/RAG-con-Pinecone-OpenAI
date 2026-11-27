"""Shared fixtures and fakes. All tests run offline: no API keys, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from config import Settings  # noqa: E402

ENV_VARS = (
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "GOOGLE_API_KEY",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "PINECONE_NAMESPACE",
    "GEMINI_CHAT_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "RAG_TEMPERATURE",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_TOP_K",
    "RAG_SCORE_THRESHOLD",
    "RAG_MAX_QUESTION_CHARS",
    "RAG_MAX_CONTEXT_CHARS",
    "RAG_MAX_RETRIES",
    "RAG_RETRY_BASE_DELAY",
    "RAG_REQUEST_TIMEOUT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove project env vars and disable .env loading so a real .env never leaks in."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    # Other modules may import load_dotenv directly; neutralise those too if loaded.
    for mod_name in ("rag_chain", "main", "ingest"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "load_dotenv"):
            monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: False)
    yield


def make_settings(**overrides) -> Settings:
    """Settings with dummy credentials and instant retries."""
    values = dict(
        pinecone_api_key="x",
        pinecone_index_name="idx",
        google_api_key="y",
        score_threshold=0.5,
        max_retries=2,
        retry_base_delay=0,
    )
    values.update(overrides)
    return Settings(**values)


def make_doc(content: str, source: str = "data/mma_rules_and_judging.md", chunk_index: int = 0):
    from langchain_core.documents import Document

    return Document(page_content=content, metadata={"source": source, "chunk_index": chunk_index})


class FakeVectorStore:
    """Records similarity_search_with_score calls and returns canned results."""

    def __init__(self, results=None, error: Exception | None = None):
        self.results = list(results or [])
        self.error = error
        self.calls: list[dict] = []

    def similarity_search_with_score(self, query, k=4, namespace=None, **kwargs):
        self.calls.append({"query": query, "k": k, "namespace": namespace, **kwargs})
        if self.error is not None:
            raise self.error
        return self.results[:k]


class FakeChain:
    """Minimal stand-in for an LCEL chain: `.invoke(inputs)` returns a string."""

    def __init__(self, answer: str = "A 10-8 round means dominance.", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list = []

    def invoke(self, inputs, *args, **kwargs):
        self.calls.append(inputs)
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make any real time.sleep instant so retry paths never slow the suite."""
    import time

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
