"""LLM gateway: one place that talks to Claude, with failure handling built in.

Why a gateway instead of calling ChatAnthropic in each node:
  * One circuit breaker and one chaos switch for every LLM call ("LLM down" is one fact).
  * Model tiers: `main` (Claude Sonnet 5.5) for reasoning-heavy steps, `fast`
    (Claude Haiku 5.5) for high-volume steps such as RLM batch analysis.
  * Every failure surfaces as `LLMUnavailableError`. Nodes catch it and use their own
    deterministic fallback (keyword routing, extractive answer...), so an LLM outage
    degrades answers instead of breaking the assistant.

Model notes (Claude 5.5 generation):
  * Sampling parameters (temperature/top_p) are rejected with non-default values, so
    none are sent.
  * Forced tool choice is rejected, so structured output uses native JSON-schema output
    (`method="json_schema"`), not the "force a tool call" trick.
  * Streamed chunks may carry thinking blocks; only text is forwarded to the user.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.errors import LLMUnavailableError
from app.logging_setup import get_logger
from app.resilience import BREAKERS, chaos

log = get_logger(__name__)
M = TypeVar("M", bound=BaseModel)


class LLMGateway:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._models: dict[str, ChatAnthropic] = {}

    @property
    def configured(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    def model(self, tier: str = "main") -> ChatAnthropic:
        if tier not in self._models:
            s = self.settings
            self._models[tier] = ChatAnthropic(
                model=s.llm_model if tier == "main" else s.llm_fast_model,
                api_key=s.anthropic_api_key,
                max_tokens=s.llm_max_tokens,
                timeout=s.llm_timeout_s,
                max_retries=s.llm_max_retries,
            )
        return self._models[tier]

    def _check_available(self) -> None:
        if not self.configured:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")
        if chaos.llm_fail:
            raise LLMUnavailableError("LLM failure injected (chaos mode)")
        if BREAKERS["llm"].is_open:
            raise LLMUnavailableError("LLM circuit open after repeated failures")

    async def _guarded(self, call: Callable[[], Awaitable], what: str):
        self._check_available()
        try:
            result = await call()
        except LLMUnavailableError:
            raise
        except Exception as exc:  # provider SDK errors: auth, overload, timeout, refusal...
            BREAKERS["llm"].record_failure()
            log.warning("llm_call_failed", call=what, error=f"{type(exc).__name__}: {exc}")
            raise LLMUnavailableError(f"{what}: {type(exc).__name__}") from exc
        BREAKERS["llm"].record_success()
        return result

    @staticmethod
    def _messages(system: str, user: str | list[BaseMessage]) -> list[BaseMessage]:
        body = [HumanMessage(user)] if isinstance(user, str) else user
        return [SystemMessage(system), *body]

    async def structured(self, schema: type[M], system: str, user: str | list[BaseMessage],
                         *, tier: str = "main", run_name: str = "structured") -> M:
        """Return a validated pydantic object (native JSON-schema output)."""
        runnable = self.model(tier).with_structured_output(schema, method="json_schema")
        msgs = self._messages(system, user)
        return await self._guarded(
            lambda: runnable.ainvoke(msgs, config={"run_name": run_name}), run_name)

    async def text(self, system: str, user: str | list[BaseMessage], *, tier: str = "main",
                   run_name: str = "text") -> str:
        msgs = self._messages(system, user)
        reply = await self._guarded(
            lambda: self.model(tier).ainvoke(msgs, config={"run_name": run_name}), run_name)
        return reply.text

    async def tool_calls(self, system: str, user: str | list[BaseMessage], tools: list[dict],
                         *, run_name: str = "tool_selection") -> list[dict]:
        """Let the model choose tool calls (tool_choice=auto). Returns [{name, args}]."""
        bound = self.model("main").bind_tools(tools)
        msgs = self._messages(system, user)
        reply = await self._guarded(
            lambda: bound.ainvoke(msgs, config={"run_name": run_name}), run_name)
        return [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]

    async def stream(self, system: str, user: str | list[BaseMessage],
                     on_token: Callable[[str], None], *, tier: str = "main",
                     run_name: str = "stream") -> str:
        """Stream text tokens to `on_token`; return the full text."""
        msgs = self._messages(system, user)

        async def run() -> str:
            parts: list[str] = []
            async for chunk in self.model(tier).astream(msgs, config={"run_name": run_name}):
                token = chunk.text
                if token:
                    parts.append(token)
                    on_token(token)
            return "".join(parts)

        return await self._guarded(run, run_name)


_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
