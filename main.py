"""CLI entrypoint for querying the MMA/UFC RAG system (Gemini + Pinecone).

Exit codes: 0 ok, 1 configuration/index/upstream/unexpected error,
2 invalid question (single-question mode), 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from config import Settings, load_settings
from errors import InvalidInputError, RAGError
from rag_chain import RAGPipeline, answer_question, build_rag_pipeline
from validation import validate_question

logger = logging.getLogger("main")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2
EXIT_INTERRUPTED = 130

_OUR_LOGGERS = ("main", "rag_chain", "retry", "config", "validation", "ingest")
_QUIT_WORDS = {"exit", "quit", "salir"}


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for name in _OUR_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG if debug else logging.WARNING)


def _snippet(text: str, limit: int = 140) -> str:
    snippet = " ".join(text.split())
    return snippet if len(snippet) <= limit else snippet[: limit - 3] + "..."


def format_sources(result: dict[str, Any]) -> str:
    """Readable source list: ``- data/x.md (chunk 3, score 0.812): <snippet>``."""
    sources = result.get("sources") or []
    documents = result.get("documents") or []
    if not sources:
        return "- No documents were retrieved."
    lines: list[str] = []
    for i, src in enumerate(sources):
        text = documents[i].page_content if i < len(documents) else ""
        score = src.get("score")
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        chunk = src.get("chunk_index")
        chunk_txt = "n/a" if chunk is None else chunk
        lines.append(f"- {src.get('source', 'unknown')} (chunk {chunk_txt}, score {score_txt}): {_snippet(text)}")
    return "\n".join(lines)


def format_result(result: dict[str, Any], threshold: float | None = None) -> str:
    parts = ["Answer:", result["answer"]]
    if result.get("abstained") and result.get("abstain_reason") == "low_score":
        top = result.get("top_score")
        top_txt = f"{top:.2f}" if isinstance(top, (int, float)) else "n/a"
        thr_txt = f"{threshold:.2f}" if threshold is not None else "?"
        parts.append(f"(Abstención: score máximo {top_txt} < umbral {thr_txt})")
    elif result.get("abstained"):
        parts.append("(Abstención: el modelo indicó que el contexto es insuficiente)")
    parts.extend(["", "Sources:", format_sources(result)])
    return "\n".join(parts)


def print_result(result: dict[str, Any], threshold: float | None = None) -> None:
    print()
    print(format_result(result, threshold))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask questions to the MMA/UFC RAG system (Gemini + Pinecone)."
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Ask one question and exit. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Number of retrieved chunks (default: RAG_TOP_K)."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Minimum top-1 similarity to answer (default: RAG_SCORE_THRESHOLD).",
    )
    parser.add_argument("--namespace", type=str, default=None, help="Pinecone namespace.")
    parser.add_argument("--debug", action="store_true", help="Verbose logs and tracebacks.")
    return parser


def _full_settings(args: argparse.Namespace) -> Settings:
    return load_settings(
        top_k=args.k, score_threshold=args.threshold, namespace=args.namespace
    )


def _run_single(args: argparse.Namespace) -> int:
    # Validate first, with limits only (no API keys needed) so bad input never hits the network.
    limits = load_settings(require_keys=False)
    try:
        question = validate_question(args.question, limits.max_question_chars)
    except InvalidInputError as exc:
        print(f"Entrada inválida: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    settings = _full_settings(args)
    pipeline = build_rag_pipeline(settings)
    result = answer_question(question, pipeline=pipeline)
    print_result(result, settings.score_threshold)
    return EXIT_OK


def _run_interactive(args: argparse.Namespace) -> int:
    settings = _full_settings(args)
    pipeline: RAGPipeline = build_rag_pipeline(settings)
    print("MMA/UFC RAG CLI (Gemini + Pinecone)")
    print("Escribe una pregunta, o 'salir' / 'exit' para terminar.")

    while True:
        try:
            raw = input("\nQuestion > ")
        except EOFError:
            print("\nExiting.")
            return EXIT_OK

        if raw.strip().lower() in _QUIT_WORDS:
            print("Exiting.")
            return EXIT_OK

        try:
            result = answer_question(raw, pipeline=pipeline)
        except InvalidInputError as exc:
            print(f"Entrada inválida: {exc}")
            continue
        except RAGError as exc:
            # Keep the session alive on transient upstream failures.
            if args.debug:
                logger.exception("Question failed")
            print(f"Error: {exc}")
            continue
        print_result(result, settings.score_threshold)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run single-question or interactive mode."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.debug)

    try:
        if args.k is not None and args.k < 1:
            print("Error: --k must be at least 1.", file=sys.stderr)
            return EXIT_INVALID_INPUT
        if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
            print("Error: --threshold must be between 0 and 1.", file=sys.stderr)
            return EXIT_INVALID_INPUT

        if args.question is not None:
            return _run_single(args)
        return _run_interactive(args)
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except InvalidInputError as exc:
        print(f"Entrada inválida: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except RAGError as exc:
        if args.debug:
            logger.exception("Request failed")
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        if args.debug:
            logger.exception("Unexpected error")
        print(
            f"Error inesperado: {type(exc).__name__}: {exc}. "
            "Ejecuta con --debug para ver el detalle.",
            file=sys.stderr,
        )
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
