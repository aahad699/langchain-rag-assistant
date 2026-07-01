from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

class EmbeddingModel:
    """
    Creates a Hugging Face embedding model for converting text chunks into vectors.
    """

    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def get_embeddings(self):
        return self.embeddings