"""Custom exception types for the RAG application."""


class RAGError(Exception):
    """Base class for all application-specific errors."""


class VectorStoreNotFoundError(RAGError):
    """Raised when the FAISS index has not been built yet."""


class NoDocumentsError(RAGError):
    """Raised when there are no source documents available to ingest."""


class ConfigurationError(RAGError):
    """Raised when a required configuration value or secret is missing."""
