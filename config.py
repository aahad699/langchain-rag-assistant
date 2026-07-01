from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Gemini model
    LLM_MODEL = "gemini-2.5-flash"

    # Text splitting
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval
    TOP_K = 3

    # Paths
    DATA_DIR = "data"
    VECTOR_DB_DIR = "vectorstore"