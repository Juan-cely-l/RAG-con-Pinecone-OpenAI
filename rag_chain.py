"""Shared RAG setup for Gemini + Pinecone using LangChain."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


@dataclass
class Settings:
    """Environment-driven settings for the RAG project."""

    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    google_api_key: str
    gemini_chat_model: str
    gemini_embedding_model: str


def load_settings() -> Settings:
    """Load and validate required environment variables."""
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    missing = [
        name
        for name, value in [
            ("PINECONE_API_KEY", pinecone_api_key),
            ("PINECONE_INDEX_NAME", pinecone_index_name),
            ("GOOGLE_API_KEY", google_api_key),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Create a .env file from .env.example and fill in the values."
        )

    return Settings(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip() or "aws",
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
        google_api_key=google_api_key,
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash").strip()
        or "gemini-1.5-flash",
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
        ).strip()
        or "models/gemini-embedding-001",
    )


def get_embeddings_model(settings: Settings | None = None) -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embeddings model."""
    settings = settings or load_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )


def get_chat_model(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    settings = settings or load_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )


def _index_names(pinecone_client: Pinecone) -> set[str]:
    """Return existing Pinecone index names across SDK response formats."""
    response = pinecone_client.list_indexes()

    if hasattr(response, "names"):
        return set(response.names())

    if isinstance(response, dict):
        items = response.get("indexes") or response.get("data") or []
    else:
        items = getattr(response, "indexes", []) or []

    names: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            names.add(item["name"])
            continue
        name = getattr(item, "name", None)
        if name:
            names.add(name)
    return names


def _index_ready(describe_response: Any) -> bool:
    """Check index readiness across SDK response formats."""
    status = getattr(describe_response, "status", None)
    if isinstance(status, dict):
        return bool(status.get("ready"))
    if status is not None:
        ready = getattr(status, "ready", None)
        if ready is not None:
            return bool(ready)
    if isinstance(describe_response, dict):
        return bool((describe_response.get("status") or {}).get("ready"))
    return False


def _index_dimension(describe_response: Any) -> int | None:
    """Read the Pinecone index dimension across SDK response formats."""
    if isinstance(describe_response, dict):
        value = describe_response.get("dimension")
        return int(value) if isinstance(value, int) else value

    value = getattr(describe_response, "dimension", None)
    if value is None:
        specification = getattr(describe_response, "spec", None)
        value = getattr(specification, "dimension", None)

    return int(value) if isinstance(value, int) else value


def get_pinecone_client(settings: Settings | None = None) -> Pinecone:
    """Create the Pinecone client."""
    settings = settings or load_settings()
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_pinecone_index(
    embeddings: GoogleGenerativeAIEmbeddings, settings: Settings | None = None
) -> None:
    """Create the Pinecone index if needed with the correct embedding dimension."""
    settings = settings or load_settings()
    pinecone_client = get_pinecone_client(settings)
    expected_dimension = len(
        embeddings.embed_query("MMA and UFC sample text for dimension check.")
    )

    if settings.pinecone_index_name in _index_names(pinecone_client):
        description = pinecone_client.describe_index(settings.pinecone_index_name)
        existing_dimension = _index_dimension(description)
        if existing_dimension and existing_dimension != expected_dimension:
            raise ValueError(
                "Existing Pinecone index dimension does not match Gemini embeddings dimension. "
                f"Index '{settings.pinecone_index_name}' has dimension {existing_dimension}, "
                f"but the embedding model '{settings.gemini_embedding_model}' returned "
                f"dimension {expected_dimension}."
            )
        return

    pinecone_client.create_index(
        name=settings.pinecone_index_name,
        dimension=expected_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    timeout_seconds = 120
    start = time.time()
    while time.time() - start < timeout_seconds:
        description = pinecone_client.describe_index(settings.pinecone_index_name)
        if _index_ready(description):
            return
        time.sleep(2)

    raise TimeoutError(
        f"Pinecone index '{settings.pinecone_index_name}' was created but did not become ready "
        f"within {timeout_seconds} seconds."
    )


def get_pinecone_vector_store(
    embeddings: GoogleGenerativeAIEmbeddings, settings: Settings | None = None
) -> PineconeVectorStore:
    """Create a LangChain Pinecone vector store instance."""
    settings = settings or load_settings()
    ensure_pinecone_index(embeddings, settings)

    pinecone_client = get_pinecone_client(settings)
    index = pinecone_client.Index(settings.pinecone_index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def _format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a single context block."""
    if not documents:
        return "No relevant context was retrieved."

    sections: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "n/a")
        sections.append(
            f"[Context {i} | source={source} | chunk={chunk_index}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(sections)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant for an MMA/UFC study project.

Use only the provided context to answer the user's question.
If the context does not contain enough information, say clearly that the context is insufficient.
Do not invent facts that are not supported by the context.

Context:
{context}

Question:
{question}

Answer:"""
)


def build_rag_pipeline(top_k: int = 4) -> tuple[Any, Any]:
    """Build and return the retriever and answer chain."""
    settings = load_settings()
    embeddings = get_embeddings_model(settings)
    vector_store = get_pinecone_vector_store(embeddings, settings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    llm = get_chat_model(settings)
    answer_chain = RAG_PROMPT | llm | StrOutputParser()
    return retriever, answer_chain


def answer_question(
    question: str,
    retriever: Any | None = None,
    answer_chain: Any | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Answer one question using Pinecone retrieval + Gemini generation."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if retriever is None or answer_chain is None:
        retriever, answer_chain = build_rag_pipeline(top_k=top_k)

    documents: list[Document] = retriever.invoke(question)
    context = _format_context(documents)
    answer = answer_chain.invoke({"question": question.strip(), "context": context})

    return {
        "question": question.strip(),
        "answer": answer.strip() if isinstance(answer, str) else str(answer),
        "documents": documents,
    }
