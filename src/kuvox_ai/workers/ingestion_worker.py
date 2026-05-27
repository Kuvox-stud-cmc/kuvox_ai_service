"""Worker: consumes ingestion jobs from RabbitMQ."""

from __future__ import annotations

import asyncio

import aio_pika

from kuvox_ai.config import get_settings
from kuvox_ai.logging import get_logger
from kuvox_ai.workers._runner import run_worker

logger = get_logger(__name__)


async def handle_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    """Handle one ingestion message.

    TODO: parse into ``IngestionRequest`` and call ``IngestionService.ingest``.
    Until then we just log and ack so the queue drains in development.
    """
    async with message.process(requeue=False):
        logger.info("ingestion_worker.message", size=len(message.body))


def run() -> None:
    settings = get_settings()
    asyncio.run(
        run_worker(
            settings.queue_ingestion,
            handle_message,
            worker_name="ingestion_worker",
        )
    )


if __name__ == "__main__":
    run()
