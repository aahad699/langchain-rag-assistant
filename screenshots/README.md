# Screenshots

Drop the following image files into this folder. They are referenced by the
main [`README.md`](../README.md); once present, they render automatically on
GitHub.

| Filename | What it should show |
|----------|---------------------|
| `streamlit-chat.png` | The Streamlit UI (`streamlit run app.py`, http://localhost:8501) after asking a question — answer visible with the **📄 Sources** list underneath. This is the hero image. |
| `api-docs.png` | The FastAPI Swagger UI at http://localhost:8000/docs showing the `/`, `/health`, and `/chat` endpoints. |
| `api-chat.png` | A successful `POST /chat` — either the Swagger "Try it out" 200 response, or a terminal `curl` showing the JSON `answer` + `sources`. |
| `ingest.png` | The terminal output of `python ingest.py` (e.g. "Indexed 812 chunk(s)."). Optional but nice. |

Recommended width: ~1200–1600px. PNG preferred.
