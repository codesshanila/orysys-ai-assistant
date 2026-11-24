# One image, three roles (backend, mcp, frontend) chosen by the compose `command`.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY mcp_server ./mcp_server
COPY frontend ./frontend
COPY scripts ./scripts
COPY data/documents ./data/documents
COPY data/index ./data/index

# Run as an unprivileged user; only data/runtime (memory, checkpoints) is writable.
RUN useradd --create-home appuser && mkdir -p data/runtime && chown -R appuser data/runtime
USER appuser

EXPOSE 8000 8001 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
