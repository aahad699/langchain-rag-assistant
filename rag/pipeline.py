from rag.embeddings import EmbeddingModel
from rag.vector_db import VectorDatabase
from rag.retriever import Retriever
from rag.llm import LLM
from rag.chain import RAGChain


class RAGPipeline:
    """
    High-level interface for interacting with the RAG system.
    """

    def __init__(self):
        embeddings = EmbeddingModel().get_embeddings()

        vectorstore = VectorDatabase().load(embeddings)

        retriever = Retriever(vectorstore).get_retriever()

        llm = LLM().get_llm()

        self.chain = RAGChain(
            llm=llm,
            retriever=retriever,
        ).build()

    def ask(self, question: str) -> dict:
        """
        Ask a question and return the answer along with source metadata.
        """

        response = self.chain.invoke(
            {
                "input": question
            }
        )

        sources = []

        for document in response["context"]:
            metadata = document.metadata

            sources.append(
                {
                    "file": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                }
            )

        return {
            "answer": response["answer"],
            "sources": sources,
        }