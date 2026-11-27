"""Tests for the main.py CLI: exit codes, friendly errors, output format."""

import builtins

import pytest

import main
from conftest import make_doc
from errors import InvalidInputError, UpstreamError


@pytest.fixture
def with_keys(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "idx")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")


def _fake_result(question):
    doc = make_doc("In a 10-8 round one fighter dominates.", source="data/scoring_criteria_detail.md", chunk_index=1)
    return {
        "question": question,
        "answer": "A 10-8 round reflects overwhelming dominance.",
        "abstained": False,
        "abstain_reason": None,
        "top_score": 0.873,
        "sources": [{"source": "data/scoring_criteria_detail.md", "chunk_index": 1, "score": 0.873}],
        "documents": [doc],
        "latency_s": 0.01,
    }


@pytest.fixture
def fake_backend(monkeypatch, with_keys):
    """Replace pipeline building and answering with offline fakes."""
    asked: list[str] = []

    def fake_answer(question, pipeline=None, top_k=None):
        from validation import validate_question

        normalized = validate_question(question, 500)
        asked.append(normalized)
        return _fake_result(normalized)

    monkeypatch.setattr(main, "build_rag_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(main, "answer_question", fake_answer)
    return asked


def test_blank_question_exits_2_without_traceback(capsys):
    code = main.main(["--question", "   "])
    err = capsys.readouterr().err
    assert code == 2
    assert err.strip()
    assert "Traceback" not in err


def test_missing_env_vars_exit_1_with_friendly_message(capsys):
    code = main.main(["--question", "How are rounds scored?"])
    captured = capsys.readouterr()
    assert code == 1
    assert "PINECONE_API_KEY" in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_invalid_k_is_rejected(capsys):
    assert main.main(["--question", "How are rounds scored?", "--k", "0"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_successful_question_prints_answer_and_sources(fake_backend, capsys):
    code = main.main(["--question", "What is a 10-8 round?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "A 10-8 round reflects overwhelming dominance." in out
    assert "Sources:" in out
    assert "data/scoring_criteria_detail.md" in out
    assert "0.873" in out


def test_upstream_error_exits_1_without_traceback(monkeypatch, with_keys, capsys):
    def failing(*a, **k):
        raise UpstreamError("Gemini rate limit after 3 attempts", retryable=True)

    monkeypatch.setattr(main, "build_rag_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(main, "answer_question", failing)
    code = main.main(["--question", "What is a 10-8 round?"])
    err = capsys.readouterr().err
    assert code == 1
    assert "rate limit" in err
    assert "Traceback" not in err


def test_interactive_mode_survives_invalid_input_and_exits(fake_backend, monkeypatch, capsys):
    inputs = iter(["", "What is a 10-8 round?", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(inputs))
    code = main.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert fake_backend == ["What is a 10-8 round?"]
    assert "Sources:" in out


def test_interactive_mode_exits_cleanly_on_eof(fake_backend, monkeypatch):
    def eof(*_):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert main.main([]) == 0
