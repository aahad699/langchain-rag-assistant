import os
from contextlib import asynccontextmanager
from typing import Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline
from utils.exceptions import RAGError
from utils.logger import get_logger

load_dotenv()

logger = get_logger("api")

# Holds the (heavy) pipeline and any load-time error so the process can
# still start and report a useful /health status even if loading fails.
state: dict = {"pipeline": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting up: loading RAG pipeline...")
        state["pipeline"] = RAGPipeline()
    except RAGError as exc:
        state["error"] = str(exc)
        logger.error("Pipeline unavailable: %s", exc)
    except Exception as exc:  # noqa: BLE001 - never let startup crash the server
        state["error"] = f"Unexpected error while loading pipeline: {exc}"
        logger.exception("Unexpected error while loading pipeline")
    yield
    state["pipeline"] = None


app = FastAPI(
    title="AI Knowledge Assistant API",
    description="Ask questions about your documents using Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")


class Source(BaseModel):
    file: str
    page: Union[int, str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {"message": "AI Knowledge Assistant API", "docs": "/docs"}


@app.get("/health")
def health():
    ready = state["pipeline"] is not None
    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "error": state["error"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail=state["error"]
            or "Pipeline not ready. Build the index with `python ingest.py`.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        return state["pipeline"].ask(question)
    except RAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Error while answering question")
        raise HTTPException(
            status_code=500,
            detail="Internal error while answering the question.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
