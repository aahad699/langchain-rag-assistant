from dotenv import load_dotenv

load_dotenv()

from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    answer = pipeline.ask(question)

    print("\nAnswer:\n")

    print(answer)