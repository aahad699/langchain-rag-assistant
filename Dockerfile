# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- Environment -----------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# --- Python dependencies ---------------------------------------------------
COPY requirements.txt .

# Install CPU-only PyTorch first from the PyTorch CPU index so we don't pull
# the multi-gigabyte CUDA build, then the remaining dependencies from PyPI.
RUN pip install --no-cache-dir torch==2.12.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- Application code ------------------------------------------------------
COPY . .

# Build the FAISS index at build time. Embeddings run locally (HuggingFace),
# so no Google API key is required here. This also warms the model cache so
# the container starts ready-to-serve.
RUN python ingest.py

# --- Run as a non-root user ------------------------------------------------
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.getenv('PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(url).status==200 else 1)"

# $PORT is provided by hosts like Render; defaults to 8000 for local runs.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
