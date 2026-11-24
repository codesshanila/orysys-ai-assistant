"""Resilience primitives: timeouts, circuit breakers and failure injection.

`with_timeout`   - every external call is bounded; a hung dependency never hangs a request.
`CircuitBreaker` - after N consecutive failures a dependency is skipped for a cool-down
                   period, so one broken service does not add its timeout to every turn
                   (this is the "butterfly effect" guard: a local failure stays local).
`chaos`          - admin-controlled failure injection used in the demo to show graceful
                   degradation for LLM, vector DB, MCP and tool timeouts.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TypeVar

from app.errors import ToolTimeoutError
from app.logging_setup import get_logger

T = TypeVar("T")
log = get_logger(__name__)


async def with_timeout(coro: Awaitable[T], seconds: float, what: str) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError as exc:
        log.warning("dependency_timeout", dependency=what, timeout_s=seconds)
        raise ToolTimeoutError(f"{what} timed out after {seconds}s") from exc


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_after_s: float = 30.0
    _failures: int = 0
    _opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            # Half-open: let the next call through as a probe.
            self._opened_at = None
            self._failures = self.failure_threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning("circuit_opened", dependency=self.name, failures=self._failures)

    def snapshot(self) -> dict:
        return {"name": self.name, "open": self.is_open, "failures": self._failures}


BREAKERS: dict[str, CircuitBreaker] = {
    "llm": CircuitBreaker("llm", failure_threshold=4, reset_after_s=20),
    "vector_db": CircuitBreaker("vector_db"),
    "mcp": CircuitBreaker("mcp"),
}


@dataclass
class ChaosFlags:
    """Failure injection switches. Only administrators can change them (see API)."""

    llm_fail: bool = False
    vector_db_fail: bool = False
    mcp_fail: bool = False
    tool_slow: bool = False  # makes tools exceed their timeout

    def as_dict(self) -> dict:
        return {
            "llm_fail": self.llm_fail,
            "vector_db_fail": self.vector_db_fail,
            "mcp_fail": self.mcp_fail,
            "tool_slow": self.tool_slow,
        }


chaos = ChaosFlags()
