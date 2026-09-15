# Gate G1: one command, clean machine, no GPU, no network at run time.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RIPPLE_PROVIDER=stub \
    RIPPLE_EMBEDDER=tfidf-svd

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ripple/ ./ripple/
COPY data/ ./data/
COPY evaluation/ ./evaluation/
COPY frontend/dashboard.html ./frontend/dashboard.html
COPY tests/ ./tests/

# Build the corpus, the labels and the index AT IMAGE BUILD TIME so the
# container starts instantly and needs no network when it runs.
RUN python data/corpus/care/build_corpus.py \
 && python data/labels/build_labels.py \
 && python data/scenarios/build_scenarios.py \
 && python -c "from ripple.retrieval.index import build_and_save; \
               build_and_save('data/corpus/care', '.index', 'tfidf-svd')"

EXPOSE 8000
CMD ["uvicorn", "ripple.server:app", "--host", "0.0.0.0", "--port", "8000"]


# --- optional higher-quality retrieval path --------------------------------
# docker build --target quality -t ripple:quality .
FROM base AS quality
COPY requirements-quality.txt .
RUN pip install --no-cache-dir -r requirements-quality.txt \
 && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
               SentenceTransformer('BAAI/bge-small-en-v1.5'); \
               CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV RIPPLE_EMBEDDER=bge-small
RUN python -c "from ripple.retrieval.index import build_and_save; \
               build_and_save('data/corpus/care', '.index', 'bge-small')"
