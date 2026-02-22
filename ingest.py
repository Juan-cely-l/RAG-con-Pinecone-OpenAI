"""Ingestion script for the MMA/UFC RAG demo (local files -> Pinecone)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chain import get_embeddings_model, get_pinecone_vector_store, load_settings

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_local_documents(data_dir: Path) -> tuple[list[Document], list[Path]]:
    """Load `.md` and `.txt` files from the local data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}. Create the folder and add .md/.txt files."
        )

    file_paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not file_paths:
        raise FileNotFoundError(
            f"No .md or .txt files found in {data_dir}. Add sample files and try again."
        )

    documents: list[Document] = []
    for path in file_paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        try:
            source = str(path.relative_to(data_dir.parent))
        except ValueError:
            source = str(path)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "filename": path.name,
                },
            )
        )

    if not documents:
        raise ValueError(f"All supported files in {data_dir} are empty.")

    return documents, file_paths


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks suitable for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    """Create deterministic chunk IDs so re-ingestion updates the same records."""
    ids: list[str] = []
    for idx, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown")).replace("\\", "/")
        safe_source = source.replace("/", "_").replace(" ", "_")
        ids.append(f"{safe_source}::chunk::{idx}")
    return ids


def main() -> int:
    """Run the ingestion pipeline."""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    print(f"Starting ingestion from: {data_dir}")

    settings = load_settings()
    documents, file_paths = load_local_documents(data_dir)
    chunks = split_documents(documents)

    embeddings = get_embeddings_model(settings)
    vector_store = get_pinecone_vector_store(embeddings, settings)
    chunk_ids = build_chunk_ids(chunks)

    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    print("Ingestion completed successfully.")
    print(f"- Files loaded: {len(file_paths)}")
    print(f"- Chunks indexed: {len(chunks)}")
    print(f"- Pinecone index: {settings.pinecone_index_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error during ingestion: {exc}")
        raise SystemExit(1)
