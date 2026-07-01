from rag.loader import PDFLoader
from rag.splitter import DocumentSplitter
from rag.embeddings import EmbeddingModel
from rag.vector_db import VectorDatabase

print("Loading PDFs...")
loader = PDFLoader()
documents = loader.load_documents()

print("Splitting documents...")
splitter = DocumentSplitter()
chunks = splitter.split_documents(documents)

print("Loading embedding model...")
embedding_model = EmbeddingModel()
embeddings = embedding_model.get_embeddings()

print("Creating vector database...")
vector_db = VectorDatabase()

vectorstore = vector_db.create(
    chunks,
    embeddings,
)

print("Saving vector database...")
vector_db.save(vectorstore)

print("Done!")
print(f"Indexed {len(chunks)} chunks.")