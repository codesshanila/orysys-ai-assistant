"""Activity events: how the UI sees what the agent is doing.

Every node emits small typed events through LangGraph's custom stream
(`get_stream_writer`). The API forwards them as Server-Sent Events, and the Streamlit
activity panel renders them live: active node, tool calls, retrieval scores, memory
updates, validation results, and answer tokens.

The `@node` decorator gives every node the same behaviour without repeating it:
  * `node_start` / `node_end` events with duration,
  * a structured log line per node,
  * a LangSmith span (LangGraph already traces nodes; the name makes it readable).
Errors are *not* swallowed here. Each node decides how to degrade, because only the node
knows what a safe fallback is.
"""
from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp

from app.logging_setup import get_logger

log = get_logger(__name__)


def emit(event_type: str, **data: Any) -> None:
    """Send one activity event. Safe to call outside a graph run (it becomes a no-op)."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"type": event_type, "ts": time.time(), **data})


def node(name: str, agent: str) -> Callable:
    """Wrap an async node so it reports start/end to the activity panel and logs."""

    def decorator(fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
        @functools.wraps(fn)
        async def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            started = time.perf_counter()
            emit("node_start", node=name, agent=agent)
            log.info("node_start", node=name, agent=agent)
            try:
                update = await fn(state, *args, **kwargs)
            except GraphBubbleUp:
                # interrupt() for human approval: a pause, not a failure.
                emit("node_paused", node=name, agent=agent)
                raise
            except Exception as exc:
                emit("node_error", node=name, agent=agent, error=type(exc).__name__)
                log.exception("node_failed", node=name)
                raise
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            emit("node_end", node=name, agent=agent, elapsed_ms=elapsed_ms)
            log.info("node_end", node=name, elapsed_ms=elapsed_ms)
            return update

        return wrapper

    return decorator
