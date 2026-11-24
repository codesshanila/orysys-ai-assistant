"""Structured logging.

All log lines are JSON (or pretty console output in dev) and automatically carry the
request-scoped context bound in `bind_request_context` - request_id, user, role, session -
so a single conversation can be followed across API, graph nodes, tools and retrieval.
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    # Keep noisy third-party loggers at WARNING.
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(message)s")


def bind_request_context(**kwargs: object) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
