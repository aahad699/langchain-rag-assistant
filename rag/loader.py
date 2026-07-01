from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from config import Config

class PDFLoader:
    """
    Loads PDF files from a directory and returns LangChain Document objects.
    """

    def __init__(self, data_folder: str = Config.DATA_DIR):
        self.data_folder = Path(data_folder)

    def load_documents(self):
        documents = []

        pdf_files = list(self.data_folder.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in '{self.data_folder}'."
            )

        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)

        return documents