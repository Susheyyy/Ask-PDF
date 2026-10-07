from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .pdf_loader import load_pdf


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=64)
def open_store(digest: str) -> Chroma:
    return Chroma(
        collection_name=f"pdf_{digest[:32]}",
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def chunk_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(pages)  


def ingest_pdf(path: str | Path) -> tuple[str, bool]:
    digest = file_hash(path)
    store = open_store(digest)
    if store.get(limit=1)["ids"]:
        return digest, True
    chunks = chunk_pages(load_pdf(path))
    store.add_documents(chunks)
    return digest, False


def search(digests: list[str], query: str, k: int | None = None) -> list[tuple[Document, float]]:
    k = k or settings.top_k
    hits: list[tuple[Document, float]] = []
    for digest in digests:
        hits.extend(open_store(digest).similarity_search_with_score(query, k=k))
    hits.sort(key=lambda pair: pair[1])
    return hits[:k]