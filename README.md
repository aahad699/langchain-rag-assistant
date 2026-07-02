# 📚 AI Knowledge Assistant (RAG)

Chat with your own PDF documents using **Retrieval-Augmented Generation (RAG)**.
Ask a question in plain English and get an answer grounded in your documents —
with page-level source citations, so nothing is made up.

Built with **LangChain**, **FAISS**, local **HuggingFace embeddings**, and
**Google Gemini** as the language model. Ships with both a **FastAPI** JSON API
and a **Streamlit** chat UI.

<p align="center">
  <img src="screenshots/streamlit-chat.png" alt="AI Knowledge Assistant — chat UI answering a question with page-cited sources" width="820">
</p>

---

## How it works

```
        ┌─────────────┐   ingest.py (one-time / on document change)
PDFs ──▶│   Loader    │──▶ split ──▶ embed ──▶ FAISS index (vectorstore/)
        └─────────────┘

        ┌─────────────┐   ask a question
Query ─▶│  Retriever  │──▶ top-k chunks ──▶ Gemini ──▶ answer + sources
        └─────────────┘
```

1. **Ingest** — PDFs in `data/` are loaded, split into overlapping chunks, and
   embedded locally with `sentence-transformers/all-MiniLM-L6-v2`. The vectors
   are stored in a local **FAISS** index (`vectorstore/`).
2. **Ask** — your question is embedded, the most similar chunks are retrieved,
   and **Gemini** answers *only* from that context. If the answer isn't in the
   documents, it says so instead of hallucinating.

---

## Project structure

```
RAG-Knowledge-Assistant/
├── api.py               # FastAPI app (JSON API)
├── app.py               # Streamlit chat UI
├── ingest.py            # Build the FAISS index from data/*.pdf
├── config.py            # Central configuration (models, chunking, paths)
├── data/                # Put your PDF documents here
├── rag/                 # RAG building blocks
│   ├── loader.py        #   PDF loading
│   ├── splitter.py      #   Text chunking
│   ├── embeddings.py    #   HuggingFace embeddings
│   ├── vector_db.py     #   FAISS create / save / load
│   ├── retriever.py     #   Similarity retriever
│   ├── llm.py           #   Gemini model
│   ├── prompt.py        #   RAG prompt template
│   ├── chain.py         #   LangChain retrieval chain
│   └── pipeline.py      #   High-level RAGPipeline.ask()
├── utils/               # Logger, custom exceptions, helpers
├── tests/               # Manual smoke scripts (run individually)
├── Dockerfile           # Container image for the API
├── render.yaml          # One-click Render deployment blueprint
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- **Python 3.11** (recommended; 3.10–3.12 should also work)
- A free **Google Gemini API key** → https://aistudio.google.com/app/apikey

---

## Local setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env          # Windows: copy .env.example .env
# then edit .env and paste your GOOGLE_API_KEY

# 4. Add PDFs to data/  (a few sample papers are already included)

# 5. Build the vector index (downloads the embedding model on first run)
python ingest.py
```

> The first `ingest.py` run downloads a ~90 MB embedding model and the first
> `pip install` pulls PyTorch, so give both a few minutes.

![Building the FAISS index with ingest.py](screenshots/ingest.png)

---

## Running the app

### FastAPI (JSON API)

```bash
python api.py
# or, with autoreload during development:
uvicorn api:app --reload
```

- Interactive docs: **http://localhost:8000/docs**
- Health check: **http://localhost:8000/health**

![FastAPI Swagger UI](screenshots/api-docs.png)

**Endpoints**

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | `/`       | Basic info                               |
| GET    | `/health` | Readiness — reports if the index loaded  |
| POST   | `/chat`   | Ask a question, get answer + sources     |

**Example**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

```json
{
  "answer": "…grounded answer…",
  "sources": [
    { "file": "2508.15080v3.pdf", "page": 3 }
  ]
}
```

![POST /chat returning an answer with sources](screenshots/api-chat.png)

### Streamlit (chat UI)

```bash
streamlit run app.py
```

Opens a chat interface at **http://localhost:8501**.

---

## Docker

The image builds the FAISS index at build time (embeddings run locally, so no
API key is needed for the build) and serves the **FastAPI** app.

```bash
# Build
docker build -t rag-assistant .

# Run — the API key is provided at runtime, never baked into the image
docker run -p 8000:8000 -e GOOGLE_API_KEY=your-key rag-assistant
```

Then open http://localhost:8000/docs.

> Rebuilding PDFs: because the index is baked in at build time, add/replace
> files in `data/` and rebuild the image to re-index.

---

## Deploy to Render

1. Push this repo to **GitHub**.
2. In the [Render dashboard](https://dashboard.render.com): **New +** →
   **Blueprint**, and select your repo. Render reads `render.yaml`.
3. When prompted, set the **`GOOGLE_API_KEY`** environment variable to your key
   (it is intentionally not stored in the repo).
4. Render builds the Docker image and deploys. The service is healthy once
   `/health` returns `{"ready": true}`.

Notes:
- The **free** plan spins down when idle; the first request after a cold start
  (plus model load) can take ~30–60 s.
- The image includes both the API and Streamlit deps. To slim it for an
  API-only deploy, remove `streamlit` from `requirements.txt` and rebuild.

---

## Configuration

Edit [`config.py`](config.py):

| Setting          | Default                                      | Purpose                        |
|------------------|----------------------------------------------|--------------------------------|
| `EMBEDDING_MODEL`| `sentence-transformers/all-MiniLM-L6-v2`     | Local embedding model          |
| `LLM_MODEL`      | `gemini-2.5-flash`                           | Gemini model                   |
| `CHUNK_SIZE`     | `1000`                                        | Characters per chunk           |
| `CHUNK_OVERLAP`  | `200`                                         | Overlap between chunks         |
| `TOP_K`          | `3`                                           | Chunks retrieved per query     |
| `DATA_DIR`       | `data`                                        | PDF source folder              |
| `VECTOR_DB_DIR`  | `vectorstore`                                 | FAISS index location           |

Environment variables (`.env`): `GOOGLE_API_KEY` (required), `LOG_LEVEL`
(optional), `PORT` (optional).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `VectorStoreNotFoundError` / `/health` shows `"ready": false` | Run `python ingest.py` to build the index. |
| `No PDF files found in 'data'` | Add at least one `.pdf` to `data/`. |
| `Missing required environment variable: GOOGLE_API_KEY` | Create `.env` from `.env.example` and add your key. |
| `/chat` returns **503** | The index isn't built or the key is missing — check `/health`. |
| Answer is *"I couldn't find that information…"* | The answer isn't in your documents, or try raising `TOP_K`. |

---

## Security

- **Never commit `.env`.** It's already in `.gitignore` and `.dockerignore`, and
  the Docker image takes the key at runtime — the secret is never baked in.
- If an API key has ever been committed or shared, **rotate it** at
  https://aistudio.google.com/app/apikey.

---

## License

Personal project — use freely.
