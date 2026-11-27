"""Tests for rag_chain retrieval, context selection, abstention and index checks."""

import pytest

import rag_chain
from conftest import FakeChain, FakeVectorStore, make_doc, make_settings
from errors import IndexNotReadyError, InvalidInputError
from rag_chain import (
    NO_INFO_MESSAGE,
    RAGPipeline,
    answer_question,
    format_context,
    retrieve,
    select_context,
)


def _pipeline(results, answer="A 10-8 round shows dominance.", **settings_overrides):
    store = FakeVectorStore(results)
    chain = FakeChain(answer)
    return RAGPipeline(vector_store=store, chain=chain, settings=make_settings(**settings_overrides)), store, chain


# --------------------------------------------------------------------------- retrieve
def test_retrieve_passes_question_k_and_namespace_to_vector_store():
    docs = [(make_doc("a"), 0.9), (make_doc("b"), 0.8)]
    pipeline, store, _ = _pipeline(docs, top_k=3, namespace="rules")
    result = retrieve(pipeline, "What is a foul?")
    assert store.calls == [{"query": "What is a foul?", "k": 3, "namespace": "rules"}]
    assert [score for _, score in result] == [0.9, 0.8]


# --------------------------------------------------------------------------- select_context
def test_select_context_keeps_chunks_within_budget():
    docs = [(make_doc("a" * 40), 0.9), (make_doc("b" * 40), 0.8), (make_doc("c" * 40), 0.7)]
    selected = select_context(docs, max_chars=85)
    assert len(selected) == 2
    assert selected[0][0].page_content == "a" * 40


def test_select_context_truncates_oversized_first_chunk_without_mutating_original():
    original = make_doc("x" * 100)
    selected = select_context([(original, 0.9), (make_doc("y"), 0.8)], max_chars=30)
    assert len(selected) == 1
    assert len(selected[0][0].page_content) <= 30
    assert original.page_content == "x" * 100
    assert selected[0][0].metadata["source"] == original.metadata["source"]


def test_format_context_includes_source_names():
    docs = [
        (make_doc("Rounds are scored 10-9.", source="data/scoring_criteria_detail.md"), 0.9),
        (make_doc("Eye pokes are fouls.", source="data/unified_rules_fouls.md", chunk_index=2), 0.8),
    ]
    text = format_context(docs)
    assert "data/scoring_criteria_detail.md" in text
    assert "data/unified_rules_fouls.md" in text
    assert "Eye pokes are fouls." in text


# --------------------------------------------------------------------------- answer_question
def test_answer_question_returns_full_result_shape():
    docs = [(make_doc("A 10-8 round...", chunk_index=3), 0.91)]
    pipeline, _, chain = _pipeline(docs)
    result = answer_question("What is a 10-8 round?", pipeline=pipeline)
    expected_keys = {
        "question", "answer", "abstained", "abstain_reason",
        "top_score", "sources", "documents", "latency_s",
    }
    assert expected_keys <= set(result)
    assert result["answer"].startswith("A 10-8 round shows dominance.")
    assert "[data/mma_rules_and_judging.md]" in result["answer"]  # citation enforced (CA6)
    assert result["abstained"] is False and result["abstain_reason"] is None
    assert result["top_score"] == pytest.approx(0.91)
    assert result["sources"] == [
        {"source": "data/mma_rules_and_judging.md", "chunk_index": 3, "score": pytest.approx(0.91)}
    ]
    assert result["latency_s"] >= 0
    assert len(chain.calls) == 1


def test_answer_question_sends_context_and_normalized_question_to_chain():
    pipeline, _, chain = _pipeline([(make_doc("Rounds last five minutes."), 0.9)])
    answer_question("  How long   is a round?  ", pipeline=pipeline)
    inputs = chain.calls[0]
    assert inputs["question"] == "How long is a round?"
    assert "Rounds last five minutes." in inputs["context"]


@pytest.mark.parametrize("bad", ["", "   ", "???", "a" * 600])
def test_invalid_question_raises_before_any_api_call(bad):
    pipeline, store, chain = _pipeline([(make_doc("x"), 0.9)])
    with pytest.raises(InvalidInputError):
        answer_question(bad, pipeline=pipeline)
    assert store.calls == [] and chain.calls == []


def test_invalid_question_without_pipeline_does_not_build_one(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("build_rag_pipeline must not be called for invalid input")

    monkeypatch.setattr(rag_chain, "build_rag_pipeline", boom)
    with pytest.raises(InvalidInputError):
        answer_question("   ")


def test_low_score_abstains_without_calling_llm():
    pipeline, _, chain = _pipeline([(make_doc("Unrelated."), 0.2)])
    result = answer_question("Who won the 1850 cricket cup?", pipeline=pipeline)
    assert result["abstained"] is True
    assert result["abstain_reason"] == "low_score"
    assert result["answer"] == NO_INFO_MESSAGE
    assert chain.calls == []


def test_no_results_abstains_without_calling_llm():
    pipeline, _, chain = _pipeline([])
    result = answer_question("What is a 10-8 round?", pipeline=pipeline)
    assert result["abstained"] is True
    assert result["answer"] == NO_INFO_MESSAGE
    assert result["sources"] == []
    assert chain.calls == []


def test_llm_saying_no_info_is_marked_as_abstained():
    pipeline, _, chain = _pipeline([(make_doc("Something."), 0.9)], answer=NO_INFO_MESSAGE)
    result = answer_question("What is a 10-8 round?", pipeline=pipeline)
    assert len(chain.calls) == 1
    assert result["abstained"] is True
    assert result["abstain_reason"] == "llm_insufficient"


def test_top_k_override_is_used_for_retrieval():
    docs = [(make_doc(str(i)), 0.9 - i * 0.01) for i in range(6)]
    pipeline, store, _ = _pipeline(docs, top_k=4)
    answer_question("What is a 10-8 round?", pipeline=pipeline, top_k=2)
    assert store.calls[0]["k"] == 2


# --------------------------------------------------------------------------- build_rag_pipeline
class FakeIndex:
    def __init__(self, stats):
        self.stats = stats

    def describe_index_stats(self, *a, **k):
        return self.stats


class FakeIndexList:
    def __init__(self, names):
        self._names = names

    def names(self):
        return list(self._names)


class FakePinecone:
    def __init__(self, names, stats=None):
        self._names = names
        self._stats = stats if stats is not None else {"total_vector_count": 0, "namespaces": {}}

    def list_indexes(self):
        return FakeIndexList(self._names)

    def Index(self, name):  # noqa: N802 - mirrors the SDK
        return FakeIndex(self._stats)


def _patch_factories(monkeypatch, client):
    monkeypatch.setattr(rag_chain, "get_pinecone_client", lambda *a, **k: client)
    monkeypatch.setattr(rag_chain, "get_embeddings_model", lambda *a, **k: object())
    monkeypatch.setattr(rag_chain, "get_chat_model", lambda *a, **k: object())


def test_build_pipeline_fails_clearly_when_index_missing(monkeypatch):
    _patch_factories(monkeypatch, FakePinecone(names=[]))
    with pytest.raises(IndexNotReadyError) as exc_info:
        rag_chain.build_rag_pipeline(settings=make_settings())
    assert "ingest.py" in str(exc_info.value)


def test_build_pipeline_fails_clearly_when_index_empty(monkeypatch):
    _patch_factories(monkeypatch, FakePinecone(names=["idx"]))
    with pytest.raises(IndexNotReadyError) as exc_info:
        rag_chain.build_rag_pipeline(settings=make_settings())
    assert "ingest.py" in str(exc_info.value)


def test_build_pipeline_fails_when_requested_namespace_is_empty(monkeypatch):
    stats = {"total_vector_count": 10, "namespaces": {"other": {"vector_count": 10}}}
    _patch_factories(monkeypatch, FakePinecone(names=["idx"], stats=stats))
    with pytest.raises(IndexNotReadyError):
        rag_chain.build_rag_pipeline(settings=make_settings(), namespace="rules")


def test_ensure_citation_appends_sources_when_missing():
    from langchain_core.documents import Document
    import rag_chain
    used = [(Document(page_content="x", metadata={"source": "data/a.md"}), 0.9)]
    out = rag_chain.ensure_citation("Respuesta sin cita.", used)
    assert out.endswith("[data/a.md]")


def test_ensure_citation_keeps_answer_that_already_cites():
    from langchain_core.documents import Document
    import rag_chain
    used = [(Document(page_content="x", metadata={"source": "data/a.md"}), 0.9)]
    answer = "Es así [data/a.md]."
    assert rag_chain.ensure_citation(answer, used) == answer
