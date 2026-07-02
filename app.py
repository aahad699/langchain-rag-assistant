import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from config import Config
from rag.pipeline import RAGPipeline

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# On Streamlit Community Cloud there is no .env file — the key is provided via
# the app's Secrets. Bridge it into the environment so langchain-google-genai
# (which reads GOOGLE_API_KEY from os.environ) can find it.
try:
    if "GOOGLE_API_KEY" in st.secrets and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    # st.secrets raises if no secrets file exists (e.g. local runs without one).
    pass

st.title("📚 AI Knowledge Assistant")
st.caption("Chat with your documents using RAG.")

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "**GOOGLE_API_KEY is not set.** Add it to `.env` locally, or to the "
        "app's Secrets on Streamlit Cloud (⋮ → Settings → Secrets) as "
        "`GOOGLE_API_KEY = \"your-key\"`, then rerun."
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def load_pipeline():
    # The FAISS index is gitignored, so a fresh cloud deploy won't ship with one.
    # Build it once from the committed PDFs in data/ if it's missing.
    index_path = Path(Config.VECTOR_DB_DIR) / "index.faiss"
    if not index_path.exists():
        with st.spinner("Building the knowledge index (first run only, this can take a minute)…"):
            from ingest import main as build_index

            build_index()
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