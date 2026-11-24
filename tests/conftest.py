"""Test setup: offline mode.

Blank API keys (environment variables beat .env in pydantic-settings), so the LLM
gateway reports "unavailable" and every node uses its deterministic fallback, and
retrieval uses the local BM25 index. Tests need no network and no credentials.
"""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="orysys-tests-")
os.environ.update({
    "ANTHROPIC_API_KEY": "", "PINECONE_API_KEY": "", "LANGSMITH_API_KEY": "", "LANGSMITH_TRACING": "false",
    "MEMORY_DB_PATH": os.path.join(_tmp, "memory.sqlite"),
    "CHECKPOINT_DB_PATH": os.path.join(_tmp, "checkpoints.sqlite"),
    "MCP_SERVER_URL": "http://127.0.0.1:1/mcp",  # nothing listens here: MCP is "down" in tests
    "LOG_LEVEL": "WARNING", "JWT_SECRET": "test-secret-that-is-long-enough-for-hs256",
})

import pytest  # noqa: E402

from app.resilience import BREAKERS, chaos  # noqa: E402

USERS = {
    "viewer": {"username": "viewer", "display_name": "V", "role": "viewer", "department": "branch-operations"},
    "analyst": {"username": "analyst", "display_name": "A", "role": "analyst", "department": "payments"},
    "admin": {"username": "admin", "display_name": "Ad", "role": "admin", "department": "technology"},
}


@pytest.fixture(autouse=True)
def reset_resilience():
    """Chaos flags and circuit breakers are process-wide; isolate every test."""
    for flag in chaos.as_dict():
        setattr(chaos, flag, False)
    for breaker in BREAKERS.values():
        breaker.record_success()
    yield
