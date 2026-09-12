"""
Deus — Structured Logging Configuration

Uses structlog for JSON-formatted, structured logging.
Call setup_logging() once at startup (auto-called at import time).
"""

from __future__ import annotations

import logging
import sys

import structlog

from config.settings import settings


# Third-party loggers that print request URLs at INFO.
#
# `httpx` emits one "HTTP Request: POST https://… " line per call, and the
# Telegram Bot API puts the bot token *in the path*
# (api.telegram.org/bot<token>/getUpdates). With polling that is several lines a
# second, so a long-running worker log becomes a file whose every page leaks the
# credential — 246 MB of it on the phone, readable by anything with filesystem
# access and by anyone the log is ever sent to.
#
# WARNING keeps the failures (timeouts, connection errors) and drops the
# per-request chatter. Raising them back to INFO requires a redacting filter
# first, not just a level change.
_NOISY_THIRD_PARTY_LOGGERS = ("httpx", "httpcore", "telegram.ext", "telegram.request")


def setup_logging() -> None:
    """Configure structlog with JSON rendering and stdlib integration."""

    # Configure stdlib logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if settings.log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger."""
    return structlog.get_logger(name)


# Auto-configure on import
setup_logging()
