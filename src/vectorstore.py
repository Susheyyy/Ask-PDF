from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .pdf_loader import load_pdf

StageCallback = Callable[[str, float], None]  
EMBED_BATCH = 32


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


def warmup() -> None:
    """Load the embedding model now (downloads it on first run) so uploads aren't slow."""
    get_embeddings().embed_query("warm up")


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


def ingest_pdf(path: str | Path, on_stage: StageCallback | None = None) -> tuple[str, bool]:
    """Index a PDF. Returns (digest, was_cached). Reports progress via on_stage."""
    notify = on_stage or (lambda message, fraction: None)
    t0 = time.perf_counter()

    notify("Checking cache", 0.02)
    digest = file_hash(path)
    store = open_store(digest)
    if store.get(limit=1)["ids"]:
        notify("Loaded from cache", 1.0)
        return digest, True

    def on_page(done: int, total: int) -> None:
        notify(f"Reading page {done}/{total}", 0.05 + 0.40 * done / total)

    pages = load_pdf(path, on_progress=on_page)
    t_read = time.perf_counter()

    notify("Splitting into chunks", 0.5)
    chunks = chunk_pages(pages)
    total = len(chunks)

    for i in range(0, total, EMBED_BATCH):
        store.add_documents(chunks[i : i + EMBED_BATCH])
        done = min(i + EMBED_BATCH, total)
        notify(f"Embedding chunks {done}/{total}", 0.5 + 0.5 * done / total)

    t_end = time.perf_counter()
    print(
        f"[ingest] {Path(path).name}: read {t_read - t0:.1f}s, "
        f"embed {t_end - t_read:.1f}s ({total} chunks)"
    )
    return digest, False


def search(digests: list[str], query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Search every selected PDF and return the overall top-k (lowest cosine distance first)."""
    k = k or settings.top_k
    hits: list[tuple[Document, float]] = []
    for digest in digests:
        hits.extend(open_store(digest).similarity_search_with_score(query, k=k))
    hits.sort(key=lambda pair: pair[1])
    return hits[:k]