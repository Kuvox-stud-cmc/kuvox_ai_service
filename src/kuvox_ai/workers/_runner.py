"""Shared boilerplate for worker entry points."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable

import aio_pika

from kuvox_ai.config import get_settings
from kuvox_ai.infrastructure import RabbitMQClient
from kuvox_ai.logging import configure_logging, get_logger

MessageHandler = Callable[[aio_pika.abc.AbstractIncomingMessage], Awaitable[None]]


async def run_worker(queue_name: str, handler: MessageHandler, *, worker_name: str) -> None:
    """Connect to RabbitMQ, consume ``queue_name`` with ``handler`` until SIGTERM/SIGINT."""
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(worker_name)
    logger.info("worker.starting", queue=queue_name)

    rabbitmq = RabbitMQClient.from_settings(settings)
    await rabbitmq.connect()
    await rabbitmq.consume(queue_name, handler)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # Signal handlers aren't supported on Windows; ignore if unavailable.
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    logger.info("worker.ready")
    try:
        await stop.wait()
    finally:
        logger.info("worker.stopping")
        await rabbitmq.close()
        logger.info("worker.stopped")
