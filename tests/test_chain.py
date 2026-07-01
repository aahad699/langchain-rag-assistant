from dotenv import load_dotenv

load_dotenv()

from rag.embeddings import EmbeddingModel
from rag.vector_db import VectorDatabase
from rag.retriever import Retriever
from rag.llm import LLM
from rag.chain import RAGChain

embeddings = EmbeddingModel().get_embeddings()

vectorstore = VectorDatabase().load(embeddings)

retriever = Retriever(vectorstore).get_retriever()

llm = LLM().get_llm()

chain = RAGChain(
    llm=llm,
    retriever=retriever,
).build()

while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    response = chain.invoke(
        {
            "input": question
        }
    )

    print("\nAnswer:\n")
    print(response["answer"])