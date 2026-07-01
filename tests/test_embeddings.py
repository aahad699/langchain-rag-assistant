from rag.embeddings import EmbeddingModel

embedding_model = EmbeddingModel()

embeddings = embedding_model.get_embeddings()

vector = embeddings.embed_query(
    "What is Machine Learning?"
)

print(f"Vector length: {len(vector)}")

print(vector[:10])