from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pdfplumber
from langchain_core.documents import Document

from .config import settings


def _table_to_markdown(table: list[list[str | None]]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in table if row]
    if len(rows) < 2 or not rows[0]:
        return ""
    header, *body = rows
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        row = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _tables_markdown(page) -> str:
    if not settings.extract_tables:
        return ""
    try:
        tables = [_table_to_markdown(t) for t in page.extract_tables()]
    except Exception:
        return ""
    tables = [t for t in tables if t]
    return ("\n\n" + "\n\n".join(tables)) if tables else ""


def _ocr_page(page) -> str:
    """OCR fallback for scanned pages. Needs pytesseract + the tesseract binary."""
    if not settings.ocr_enabled:
        return ""
    try:
        import pytesseract

        image = page.to_image(resolution=200).original
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def load_pdf(
    path: str | os.PathLike,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Document]:
    """Return one Document per page that has text. Raises ValueError on bad input.

    on_progress(done_pages, total_pages) is called after each page.
    """
    path = Path(path)
    size_mb = path.stat().st_size / 1_000_000
    if size_mb > settings.max_pdf_mb:
        raise ValueError(
            f"{path.name} is {size_mb:.1f} MB (limit {settings.max_pdf_mb:.0f} MB)."
        )

    docs: list[Document] = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        if total > settings.max_pages:
            raise ValueError(f"{path.name} has {total} pages (limit {settings.max_pages}).")
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                text = _ocr_page(page)
            content = text + _tables_markdown(page)
            if content.strip():
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": path.name, "page": number},
                    )
                )
            page.flush_cache()  
            if on_progress:
                on_progress(number, total)

    if not docs:
        raise ValueError(
            f"No extractable text in {path.name}. "
            "If it is a scanned PDF, install pytesseract and the tesseract binary."
        )
    return docs