"""Client for the enterprise MCP server (employee directory, service catalog, incidents).

One short-lived MCP session per call keeps the POC simple and stateless; each call is
bounded by a timeout and guarded by the `mcp` circuit breaker, so a dead MCP server
costs one timeout, not one per turn. The chaos switches simulate an outage or a slow
server for the demo.
"""
from __future__ import annotations

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.config import get_settings
from app.errors import MCPError, ToolTimeoutError
from app.logging_setup import get_logger
from app.resilience import BREAKERS, chaos, with_timeout

log = get_logger(__name__)


async def _call(url: str, tool: str, arguments: dict) -> dict:
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
    text = "".join(getattr(block, "text", "") for block in result.content)
    if result.isError:
        raise MCPError(f"MCP tool {tool} returned an error: {text[:200]}")
    if result.structuredContent:
        return result.structuredContent
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


async def call_mcp_tool(tool: str, arguments: dict) -> dict:
    settings = get_settings()
    breaker = BREAKERS["mcp"]
    if chaos.mcp_fail:
        raise MCPError("MCP failure injected (chaos mode)")
    if breaker.is_open:
        raise MCPError("MCP circuit open after repeated failures")
    try:
        if chaos.tool_slow:
            await asyncio.sleep(settings.tool_timeout_s + 1)
        result = await with_timeout(_call(settings.mcp_server_url, tool, arguments),
                                    settings.tool_timeout_s, f"mcp:{tool}")
    except ToolTimeoutError:
        breaker.record_failure()
        raise
    except MCPError:
        breaker.record_failure()
        raise
    except Exception as exc:  # connection refused, protocol errors, ExceptionGroup from anyio
        breaker.record_failure()
        log.warning("mcp_call_failed", tool=tool, error=f"{type(exc).__name__}: {exc}")
        raise MCPError(f"MCP call {tool} failed: {type(exc).__name__}") from exc
    breaker.record_success()
    return result
