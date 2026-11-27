"""Ingestion script for the MMA/UFC RAG demo (local files -> Pinecone).

Usage:
    python ingest.py [--namespace NS] [--reset | --keep] [--chunk-size N] [--chunk-overlap N] [--debug]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, load_settings
from errors import ConfigError, RAGError
from rag_chain import (
    _pinecone_call,
    ensure_pinecone_index,
    get_embeddings_model,
    get_pinecone_client,
)
from retry import call_with_retry

logger = logging.getLogger("ingest")

SUPPORTED_EXTENSIONS = {".md", ".txt"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
UPSERT_BATCH_SIZE = 32

_DEFAULTS = Settings()


class CorpusNotFoundError(ConfigError, FileNotFoundError):
    """Data folder missing or without supported files (friendly RAGError, still a FileNotFoundError)."""


def load_local_documents(data_dir: Path) -> tuple[list[Document], list[Path]]:
    """Load `.md` and `.txt` files from the local data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise CorpusNotFoundError(
            f"Data folder not found: {data_dir}. Create the folder and add .md/.txt files."
        )

    file_paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not file_paths:
        raise CorpusNotFoundError(
            f"No .md or .txt files found in {data_dir}. Add sample files and try again."
        )

    documents: list[Document] = []
    for path in file_paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ConfigError(f"File {path} is not valid UTF-8 text.") from exc
        if not text:
            logger.warning("Skipping empty file: %s", path)
            continue

        try:
            source = path.resolve().relative_to(data_dir.resolve().parent).as_posix()
        except ValueError:
            source = path.as_posix()

        documents.append(
            Document(page_content=text, metadata={"source": source, "filename": path.name})
        )

    if not documents:
        raise ConfigError(f"All supported files in {data_dir} are empty.")

    return documents, file_paths


def split_documents(
    documents: list[Document],
    chunk_size: int = _DEFAULTS.chunk_size,
    chunk_overlap: int = _DEFAULTS.chunk_overlap,
) -> list[Document]:
    """Split documents into chunks; ``chunk_index`` is numbered per source file."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    per_source: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        idx = per_source.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        per_source[source] = idx + 1

    return chunks


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    """Deterministic IDs (source + chunk index) so re-ingestion overwrites the same records."""
    ids: list[str] = []
    for position, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown")).replace("\\", "/")
        safe_source = source.replace("/", "_").replace(" ", "_")
        idx = chunk.metadata.get("chunk_index", position)
        ids.append(f"{safe_source}::chunk::{idx}")
    return ids


def _is_namespace_not_found(exc: BaseException | None) -> bool:
    while exc is not None:
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status == 404 or "namespace not found" in str(exc).lower():
            return True
        exc = exc.__cause__
    return False


def run_ingestion(
    settings: Settings | None = None,
    namespace: str | None = None,
    reset: bool = False,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Load, chunk, embed and upsert the corpus. Creates the index if missing."""
    settings = settings or load_settings()
    if namespace is not None:
        settings = replace(settings, namespace=namespace)
    ns = settings.namespace or ""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    logger.info("Starting ingestion from: %s", data_dir)
    documents, file_paths = load_local_documents(data_dir)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    chunk_ids = build_chunk_ids(chunks)
    logger.info(
        "Loaded %d files -> %d chunks (size=%d, overlap=%d)",
        len(file_paths),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )

    embeddings = get_embeddings_model(settings)
    ensure_pinecone_index(embeddings, settings)

    pinecone_client = get_pinecone_client(settings)
    index = _pinecone_call(
        settings,
        lambda: pinecone_client.Index(settings.pinecone_index_name),
        "Pinecone connect to index",
    )

    if reset:
        logger.info("Reset: deleting all vectors in namespace '%s'", ns or "(default)")
        try:
            _pinecone_call(
                settings,
                lambda: index.delete(delete_all=True, namespace=ns),
                "Pinecone delete namespace",
            )
        except RAGError as exc:
            if not _is_namespace_not_found(exc):
                raise
            logger.info("Namespace '%s' did not exist yet; nothing to delete.", ns or "(default)")

    from langchain_pinecone import PineconeVectorStore

    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=ns or None)
    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        batch_ids = chunk_ids[start : start + UPSERT_BATCH_SIZE]
        call_with_retry(
            lambda: vector_store.add_documents(documents=batch, ids=batch_ids, namespace=ns or None),
            max_retries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            description=f"Upsert batch {start // UPSERT_BATCH_SIZE + 1}",
        )
        logger.info("Upserted %d/%d chunks", min(start + len(batch), total), total)

    summary = {
        "files": len(file_paths),
        "chunks": total,
        "index": settings.pinecone_index_name,
        "namespace": ns,
    }
    logger.info(
        "Ingestion completed: %d files, %d chunks -> index '%s' (namespace '%s')",
        summary["files"],
        summary["chunks"],
        summary["index"],
        ns or "(default)",
    )
    return summary


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s" if debug else "%(message)s",
        stream=sys.stdout,
    )
    for name in ("ingest", "rag_chain", "retry", "config", "validation"):
        logging.getLogger(name).setLevel(logging.DEBUG if debug else logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index the local MMA/UFC corpus into Pinecone.")
    parser.add_argument("--namespace", type=str, default=None, help="Pinecone namespace.")
    parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Delete all vectors in the namespace before indexing (default) so stale "
        "chunks never survive a change of chunk size or corpus.",
    )
    parser.add_argument(
        "--keep",
        dest="reset",
        action="store_false",
        help="Do not delete existing vectors (upsert only; stale chunks may remain).",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Override RAG_CHUNK_SIZE.")
    parser.add_argument(
        "--chunk-overlap", type=int, default=None, help="Override RAG_CHUNK_OVERLAP."
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logs and tracebacks.")
    args = parser.parse_args(argv)
    _configure_logging(args.debug)

    try:
        settings = load_settings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        run_ingestion(settings, namespace=args.namespace, reset=args.reset)
        return 0
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr)
        return 130
    except RAGError as exc:
        if args.debug:
            logger.exception("Ingestion failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            logger.exception("Unexpected error during ingestion")
        print(
            f"Error inesperado: {type(exc).__name__}: {exc}. "
            "Ejecuta con --debug para ver el detalle.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
