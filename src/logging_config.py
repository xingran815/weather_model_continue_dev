"""Shared logging configuration with JSON output and correlation IDs."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id_ctx.set(correlation_id)


def get_correlation_id() -> str:
    """Return the current context correlation ID."""
    return _correlation_id_ctx.get()


def clear_correlation_id() -> None:
    """Reset the context correlation ID to default."""
    _correlation_id_ctx.set("-")


class CorrelationIdFilter(logging.Filter):
    """Attach correlation ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    _SKIP_FIELDS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }

        for key, value in record.__dict__.items():
            if key not in self._SKIP_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit JSON logs to stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler_exists = any(isinstance(handler.formatter, JsonFormatter) for handler in root_logger.handlers)
    if handler_exists:
        return

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(JsonFormatter())
    stream_handler.addFilter(CorrelationIdFilter())

    root_logger.handlers = [stream_handler]
