"""
Build the FAISS vector index from the PDFs in the data directory.

Run this once before starting the API or UI, and again whenever the
documents in `data/` change:

    python ingest.py

Embeddings are computed locally (HuggingFace), so this step does NOT
require a Google API key.
"""

from rag.loader import PDFLoader
from rag.splitter import DocumentSplitter
from rag.embeddings import EmbeddingModel
from rag.vector_db import VectorDatabase

from utils.exceptions import NoDocumentsError
from utils.logger import get_logger

logger = get_logger("ingest")


def main() -> None:
    logger.info("Loading PDFs...")
    documents = PDFLoader().load_documents()
    logger.info("Loaded %d page(s).", len(documents))

    logger.info("Splitting documents into chunks...")
    chunks = DocumentSplitter().split_documents(documents)
    if not chunks:
        raise NoDocumentsError("No text could be extracted from the PDFs.")
    logger.info("Created %d chunk(s).", len(chunks))

    logger.info("Loading embedding model (first run downloads it)...")
    embeddings = EmbeddingModel().get_embeddings()

    logger.info("Building FAISS index...")
    vector_db = VectorDatabase()
    vectorstore = vector_db.create(chunks, embeddings)

    logger.info("Saving index to disk...")
    vector_db.save(vectorstore)

    logger.info("Done. Indexed %d chunk(s).", len(chunks))


if __name__ == "__main__":
    main()
