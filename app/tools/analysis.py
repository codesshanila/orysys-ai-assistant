"""Python analysis tool: structured analysis over data the user is allowed to see.

The model writes a few lines of Python; it runs in the AST sandbox
(app/security/sandbox.py) with exactly two data inputs:
  * `catalog`  - document rows filtered by the user's role (no document text),
  * `evidence` - the chunks already retrieved this turn (already role-filtered).
So the analysis can never reach data that retrieval would not have returned.
"""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.errors import SandboxError, ToolTimeoutError
from app.retrieval.documents import load_catalog
from app.security.rbac import allowed_access_levels
from app.security.sandbox import run_sandboxed


def role_catalog(role: str) -> list[dict]:
    access = allowed_access_levels(role)
    return [row for row in load_catalog(get_settings().catalog_path) if row["access_level"] in access]


def _jsonable(value, depth: int = 0):
    """Keep results small and serialisable for the UI, traces and the LLM."""
    if depth > 4:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(value)[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:500]
    return str(value)[:200]


async def run_analysis(code: str, role: str, evidence: list[dict]) -> dict:
    injected = {"catalog": role_catalog(role),
                "evidence": [{k: e.get(k) for k in ("chunk_id", "doc_id", "title", "section", "text",
                                                    "document_type", "created_date")} for e in evidence]}
    try:
        env = await asyncio.wait_for(asyncio.to_thread(run_sandboxed, code, injected),
                                     timeout=get_settings().sandbox_timeout_s)
    except asyncio.TimeoutError as exc:
        raise ToolTimeoutError("analysis exceeded the sandbox time limit") from exc
    if "result" not in env:
        raise SandboxError("analysis code must assign a variable named `result`")
    return {"result": _jsonable(env["result"])}
