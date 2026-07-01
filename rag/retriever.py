from langchain_core.vectorstores import VectorStoreRetriever


class Retriever:
    """
    Creates a retriever from a vector database.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_retriever(self, k: int = 3) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )