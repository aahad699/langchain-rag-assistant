from rag.embeddings import EmbeddingModel
from rag.vector_db import VectorDatabase
from rag.retriever import Retriever

embedding_model = EmbeddingModel()
embeddings = embedding_model.get_embeddings()

vector_db = VectorDatabase()

vectorstore = vector_db.load(embeddings)

retriever = Retriever(vectorstore).get_retriever(k=3)

query = input("Ask a question: ")

results = retriever.invoke(query)

print("\nRetrieved Chunks:\n")

for i, doc in enumerate(results, start=1):
    print("=" * 60)
    print(f"Chunk {i}")
    print("-" * 60)
    print(doc.page_content[:500])
    print("\nMetadata:")
    print(doc.metadata)
    print()