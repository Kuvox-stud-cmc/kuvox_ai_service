"""Structured logging configuration.

Uses structlog over stdlib logging. JSON output in non-development environments
so logs are easy to ship; pretty console output in development.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from kuvox_ai.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure stdlib logging and structlog. Idempotent."""
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_development:
        renderer: Processor = structlog.dev.ConsoleRenderer(colors=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            _drop_color_message_key,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    # Quiet down noisy third-party loggers in development.
    for noisy in ("uvicorn.access", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _drop_color_message_key(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Uvicorn injects ``color_message``; we don't want it in structured output."""
    event_dict.pop("color_message", None)
    return event_dict


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.stdlib.get_logger(name).bind(**initial_values)
