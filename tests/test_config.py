"""Tests for config.load_settings."""

import dataclasses

import pytest

from config import load_settings
from errors import ConfigError


def _set_keys(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "mma-index")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")


def test_missing_keys_are_all_listed_in_error():
    with pytest.raises(ConfigError) as exc_info:
        load_settings(use_dotenv=False)
    message = str(exc_info.value)
    for name in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "GOOGLE_API_KEY"):
        assert name in message


def test_only_the_missing_key_is_reported(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "mma-index")
    with pytest.raises(ConfigError) as exc_info:
        load_settings(use_dotenv=False)
    assert "GOOGLE_API_KEY" in str(exc_info.value)
    assert "PINECONE_API_KEY" not in str(exc_info.value)


def test_require_keys_false_allows_missing_credentials():
    settings = load_settings(require_keys=False, use_dotenv=False)
    assert settings.pinecone_api_key == ""
    assert settings.top_k >= 1


def test_env_overrides_are_parsed_with_correct_types(monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.65")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
    settings = load_settings(use_dotenv=False)
    assert settings.top_k == 7
    assert settings.score_threshold == pytest.approx(0.65)
    assert (settings.chunk_size, settings.chunk_overlap) == (500, 50)
    assert settings.pinecone_index_name == "mma-index"


def test_explicit_overrides_win_over_env(monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.setenv("RAG_TOP_K", "7")
    assert load_settings(use_dotenv=False, top_k=2).top_k == 2


def test_settings_are_frozen(monkeypatch):
    _set_keys(monkeypatch)
    settings = load_settings(use_dotenv=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.top_k = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "env, value",
    [
        ("RAG_TOP_K", "four"),
        ("RAG_CHUNK_SIZE", "abc"),
        ("RAG_SCORE_THRESHOLD", "1.5"),
        ("RAG_TOP_K", "0"),
    ],
)
def test_invalid_values_raise_config_error(monkeypatch, env, value):
    _set_keys(monkeypatch)
    monkeypatch.setenv(env, value)
    with pytest.raises(ConfigError):
        load_settings(use_dotenv=False)


def test_overlap_not_smaller_than_chunk_size_is_rejected(monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")
    with pytest.raises(ConfigError):
        load_settings(use_dotenv=False)
