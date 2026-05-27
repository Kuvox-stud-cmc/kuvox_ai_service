"""Worker: consumes sandbox execution jobs from RabbitMQ."""

from __future__ import annotations

import asyncio

import aio_pika

from kuvox_ai.config import get_settings
from kuvox_ai.logging import get_logger
from kuvox_ai.workers._runner import run_worker

logger = get_logger(__name__)


async def handle_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    """Handle one sandbox job message.

    TODO: parse into ``SandboxJob`` and call ``SandboxService.execute``.
    """
    async with message.process(requeue=False):
        logger.info("sandbox_worker.message", size=len(message.body))


def run() -> None:
    settings = get_settings()
    asyncio.run(
        run_worker(
            settings.queue_sandbox,
            handle_message,
            worker_name="sandbox_worker",
        )
    )


if __name__ == "__main__":
    run()
