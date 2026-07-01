from pathlib import Path

from langchain_community.vectorstores import FAISS
from config import Config


class VectorDatabase:
    """
    Handles creation, saving, and loading of the FAISS vector database.
    """

    def __init__(self, db_path: str = Config.VECTOR_DB_DIR):
        self.db_path = Path(db_path)

    def create(self, documents, embeddings):
        """
        Create a FAISS index from document chunks.
        """
        return FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )

    def save(self, vectorstore):
        """
        Save the FAISS index locally.
        """
        vectorstore.save_local(str(self.db_path))

    def load(self, embeddings):
        """
        Load an existing FAISS index.
        """
        return FAISS.load_local(
            str(self.db_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )