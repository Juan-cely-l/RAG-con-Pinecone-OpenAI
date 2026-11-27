"""Tests for ingest.py loading, chunking and ID generation (no Pinecone calls)."""

from pathlib import Path

import pytest

from errors import RAGError
from ingest import build_chunk_ids, load_local_documents, split_documents

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _docs():
    from langchain_core.documents import Document

    long_text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    return [
        Document(page_content=long_text, metadata={"source": "data/a.md"}),
        Document(page_content=long_text, metadata={"source": "data/b file.md"}),
    ]


def test_split_documents_respects_chunk_size_and_sets_chunk_index():
    chunks = split_documents(_docs(), chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 2
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(isinstance(c.metadata["chunk_index"], int) for c in chunks)
    assert {c.metadata["source"] for c in chunks} == {"data/a.md", "data/b file.md"}


def test_smaller_chunk_size_produces_more_chunks():
    big = split_documents(_docs(), chunk_size=800, chunk_overlap=0)
    small = split_documents(_docs(), chunk_size=200, chunk_overlap=0)
    assert len(small) > len(big)


def test_chunk_ids_are_deterministic_and_unique():
    first = build_chunk_ids(split_documents(_docs(), chunk_size=300, chunk_overlap=30))
    second = build_chunk_ids(split_documents(_docs(), chunk_size=300, chunk_overlap=30))
    assert first == second
    assert len(set(first)) == len(first)
    assert all(" " not in chunk_id and "/" not in chunk_id for chunk_id in first)


def test_empty_directory_raises(tmp_path):
    with pytest.raises((FileNotFoundError, RAGError)):
        load_local_documents(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises((FileNotFoundError, RAGError)):
        load_local_documents(tmp_path / "does-not-exist")


def test_loads_only_supported_files_from_tmp_dir(tmp_path):
    corpus = tmp_path / "data"
    corpus.mkdir()
    (corpus / "rules.md").write_text("# Rules\nRounds last five minutes.", encoding="utf-8")
    (corpus / "notes.txt").write_text("Judges score each round.", encoding="utf-8")
    (corpus / "image.png").write_bytes(b"\x89PNG")
    documents, paths = load_local_documents(corpus)
    assert len(documents) == 2 and len(paths) == 2
    assert {d.metadata["source"] for d in documents} == {"data/rules.md", "data/notes.txt"}


def test_real_corpus_loads_seven_files_with_data_prefix():
    documents, paths = load_local_documents(DATA_DIR)
    assert len(paths) == 7
    assert len(documents) == 7
    assert all(d.metadata["source"].startswith("data/") for d in documents)
