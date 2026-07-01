from dotenv import load_dotenv
from rag.llm import LLM

load_dotenv()

llm = LLM().get_llm()

response = llm.invoke("Say hello in one sentence.")

print(response.content)