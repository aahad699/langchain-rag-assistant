"""Small helper utilities shared across the RAG application."""

import os

from utils.exceptions import ConfigurationError


def require_env(name: str) -> str:
    """
    Return the value of an environment variable or raise ``ConfigurationError``.
    """
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value


def format_sources(documents) -> list[dict]:
    """
    Normalise and de-duplicate source metadata from retrieved documents.

    - Collapses full file paths to just the file name.
    - Converts pypdf's 0-indexed page numbers to human-friendly 1-indexed.
    - Removes duplicate (file, page) pairs while preserving order.
    """
    seen = set()
    sources: list[dict] = []

    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}

        file = os.path.basename(str(metadata.get("source", "Unknown")))

        page = metadata.get("page", "Unknown")
        if isinstance(page, int):
            page = page + 1  # pypdf pages are 0-indexed

        key = (file, page)
        if key in seen:
            continue

        seen.add(key)
        sources.append({"file": file, "page": page})

    return sources
