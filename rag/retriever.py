from langchain_core.vectorstores import VectorStoreRetriever

from config import Config


class Retriever:
    """
    Creates a retriever from a vector database.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_retriever(self, k: int = Config.TOP_K) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )