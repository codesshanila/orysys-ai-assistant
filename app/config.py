"""Central configuration.

Every tunable lives here and is read from environment variables (or a local .env file),
so the same image runs locally, in Docker Compose and in CI without code changes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # --- Identity / brand ---------------------------------------------------------
    assistant_name: str = "Commercial Bank Knowledge Assistant"
    company_name: str = "Commercial Bank"

    # --- LLM ----------------------------------------------------------------------
    anthropic_api_key: str = ""
    # Reasoning-heavy steps (supervisor, research planning, final answer).
    llm_model: str = "claude-sonnet-5-5"
    # Cheap, fast steps (batch analysis in the RLM map phase, validation judge).
    llm_fast_model: str = "claude-haiku-5-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- Pinecone -----------------------------------------------------------------
    pinecone_api_key: str = ""
    pinecone_index: str = "orysys-knowledge"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embed_model: str = "llama-text-embed-v2"
    embed_dim: int = 1024
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_enabled: bool = True

    # --- Retrieval ----------------------------------------------------------------
    hybrid_alpha: float = Field(0.6, ge=0.0, le=1.0, description="1.0 = dense only, 0.0 = sparse only")
    retrieval_top_k: int = 8
    retrieval_candidates: int = 20
    retrieval_timeout_s: float = 15.0

    # --- RLM research agent -------------------------------------------------------
    rlm_batch_size: int = 4
    rlm_max_depth: int = 2
    rlm_max_docs: int = 24
    rlm_max_parallel: int = 4

    # --- Tools / MCP --------------------------------------------------------------
    mcp_server_url: str = "http://localhost:8001/mcp"
    tool_timeout_s: float = 10.0
    sandbox_timeout_s: float = 5.0

    # --- Auth / rate limiting -----------------------------------------------------
    jwt_secret: str = "change-me-in-production-please-32b"
    jwt_ttl_minutes: int = 120
    rate_limit_capacity: int = 10          # burst size per user
    rate_limit_refill_per_minute: float = 6  # sustained requests per minute per user

    # --- Memory -------------------------------------------------------------------
    memory_db_path: str = str(ROOT_DIR / "data" / "runtime" / "memory.sqlite")
    memory_window_messages: int = 12  # recent messages kept verbatim before summarising

    # --- Paths --------------------------------------------------------------------
    documents_dir: str = str(ROOT_DIR / "data" / "documents")
    chunk_store_path: str = str(ROOT_DIR / "data" / "index" / "chunks.jsonl")
    catalog_path: str = str(ROOT_DIR / "data" / "index" / "catalog.json")
    bm25_stats_path: str = str(ROOT_DIR / "data" / "index" / "bm25_stats.json")

    # --- Observability ------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    langsmith_project: str = "orysys-ai-assistant"

    # --- Frontend -----------------------------------------------------------------
    backend_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
