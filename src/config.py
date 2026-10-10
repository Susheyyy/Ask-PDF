from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "1.0"))

    llm_thinking: str = os.getenv("LLM_THINKING", "low").strip().lower()

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "6"))

    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma"))

    max_pdf_mb: float = float(os.getenv("MAX_PDF_MB", "25"))
    max_pages: int = int(os.getenv("MAX_PAGES", "300"))

    ocr_enabled: bool = _bool("OCR_ENABLED", True)

    extract_tables: bool = _bool("EXTRACT_TABLES", False)
    share: bool = _bool("GRADIO_SHARE", False)


settings = Settings()


def require_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to the .env file in the project root "
            "(next to app.py) and restart the app."
        )