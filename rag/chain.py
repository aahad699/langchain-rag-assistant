from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from rag.prompt import get_prompt


class RAGChain:
    """
    Creates the complete Retrieval-Augmented Generation chain.
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def build(self):
        prompt = get_prompt()

        document_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
        )

        retrieval_chain = create_retrieval_chain(
            self.retriever,
            document_chain,
        )

        return retrieval_chain