FROM python:3.13-slim

# System deps for unstructured[pdf]
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY src/ src/

# Install project with uv (no venv, system install)
RUN uv pip install --system -e .

# Upload dir
RUN mkdir -p /data/uploads

EXPOSE 8000

CMD ["uvicorn", "elser_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
