from rag.embeddings import EmbeddingModel
from rag.vector_db import VectorDatabase
from rag.retriever import Retriever
from rag.llm import LLM
from rag.chain import RAGChain

from utils.helpers import format_sources
from utils.logger import get_logger

logger = get_logger("rag.pipeline")


class RAGPipeline:
    """
    High-level interface for interacting with the RAG system.
    """

    def __init__(self):
        logger.info("Loading embedding model...")
        embeddings = EmbeddingModel().get_embeddings()

        logger.info("Loading vector store...")
        vectorstore = VectorDatabase().load(embeddings)

        retriever = Retriever(vectorstore).get_retriever()

        logger.info("Initialising LLM...")
        llm = LLM().get_llm()

        self.chain = RAGChain(
            llm=llm,
            retriever=retriever,
        ).build()

        logger.info("RAG pipeline ready.")

    def ask(self, question: str) -> dict:
        """
        Ask a question and return the answer along with source metadata.
        """

        response = self.chain.invoke(
            {
                "input": question
            }
        )

        return {
            "answer": response["answer"],
            "sources": format_sources(response["context"]),
        }
