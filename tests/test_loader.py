from rag.loader import PDFLoader
from rag.splitter import DocumentSplitter

loader = PDFLoader()
documents = loader.load_documents()

print(f"Pages loaded: {len(documents)}")

splitter = DocumentSplitter()

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

print("\nFirst chunk:\n")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)