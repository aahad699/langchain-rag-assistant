from dotenv import load_dotenv
import streamlit as st

from rag.pipeline import RAGPipeline

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Knowledge Assistant")
st.caption("Chat with your documents using RAG.")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# -------------------------
# Chat History
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                st.markdown("**📄 Sources**")

                for source in sources:

                    st.caption(
                        f"{source['file']} (Page {source['page']})"
                    )

# User input

question = st.chat_input("Ask anything about your documents...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = pipeline.ask(question)

        st.markdown(result["answer"])

        if result["sources"]:

            st.markdown("**📄 Sources**")

            for source in result["sources"]:

                st.caption(
                    f"{source['file']} (Page {source['page']})"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )