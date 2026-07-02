from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class DocumentSplitter:
    """
    Splits LangChain Document objects into smaller overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)