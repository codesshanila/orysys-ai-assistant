"""Typed errors.

Each dependency failure has its own exception so the graph can degrade per dependency
(e.g. Pinecone down -> local BM25; MCP down -> answer without live data) instead of
failing the whole request.
"""
from __future__ import annotations


class AssistantError(Exception):
    """Base class. `user_message` is safe to show; the exception text is for logs."""

    user_message = "Something went wrong while processing your request."
    status_code = 500

    def __init__(self, detail: str = "", *, user_message: str | None = None):
        super().__init__(detail or self.user_message)
        if user_message:
            self.user_message = user_message


class LLMUnavailableError(AssistantError):
    user_message = "The language model is temporarily unavailable. Please try again shortly."
    status_code = 503


class VectorStoreError(AssistantError):
    user_message = "The knowledge index is temporarily unavailable; results may be limited."
    status_code = 503


class MCPError(AssistantError):
    user_message = "The enterprise data service is temporarily unavailable."
    status_code = 503


class ToolTimeoutError(AssistantError):
    user_message = "A tool took too long to respond."
    status_code = 504


class ToolPermissionError(AssistantError):
    user_message = "You do not have permission to use that capability."
    status_code = 403


class ToolValidationError(AssistantError):
    user_message = "The tool request was invalid."
    status_code = 400


class SandboxError(AssistantError):
    user_message = "The analysis code was rejected by the safety sandbox."
    status_code = 400


class GuardrailViolation(AssistantError):
    user_message = "This request can't be processed because it conflicts with usage policy."
    status_code = 400


class RateLimitExceeded(AssistantError):
    user_message = "Too many requests. Please wait a moment and try again."
    status_code = 429

    def __init__(self, retry_after_s: float):
        super().__init__(f"rate limit exceeded, retry after {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s
