from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


class LLM:
    """
    Creates and returns the configured Gemini LLM.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=0,
        )

    def get_llm(self):
        return self.llm