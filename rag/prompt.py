from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    """
    Returns the prompt template used by the RAG chain.
    """

    return ChatPromptTemplate.from_template(
        """
You are an AI assistant that answers questions ONLY using the provided context.

Rules:
1. Answer only from the provided context.
2. If the answer is not in the context, say:
   "I couldn't find that information in the uploaded documents."
3. Do not make up facts.
4. Be clear and concise.

Context:
{context}

Question:
{input}

Answer:
"""
    )