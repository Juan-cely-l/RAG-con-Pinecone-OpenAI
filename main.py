"""CLI entrypoint for querying the MMA/UFC RAG system."""

from __future__ import annotations

import argparse

from rag_chain import answer_question, build_rag_pipeline


def format_sources(documents: list) -> str:
    """Return a readable source list for the CLI."""
    if not documents:
        return "- No documents were retrieved."

    lines: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "n/a")
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        lines.append(f"- {source} (chunk {chunk_index}): {snippet}")
    return "\n".join(lines)


def print_result(result: dict) -> None:
    """Print a single RAG response in the terminal."""
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    print(format_sources(result["documents"]))


def interactive_loop(top_k: int) -> int:
    """Run an interactive question loop."""
    retriever, answer_chain = build_rag_pipeline(top_k=top_k)
    print("MMA/UFC RAG CLI (Gemini + Pinecone)")
    print("Type a question, or type 'exit' to quit.")

    while True:
        try:
            question = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            return 0

        if not question:
            print("Please enter a question.")
            continue

        result = answer_question(
            question,
            retriever=retriever,
            answer_chain=answer_chain,
        )
        print_result(result)


def main() -> int:
    """Parse CLI arguments and run single-question or interactive mode."""
    parser = argparse.ArgumentParser(
        description="Ask questions to the MMA/UFC RAG system (Gemini + Pinecone)."
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Ask one question and exit. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of retrieved chunks (default: 4).",
    )
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    if args.question:
        result = answer_question(args.question, top_k=args.k)
        print_result(result)
        return 0

    return interactive_loop(top_k=args.k)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
