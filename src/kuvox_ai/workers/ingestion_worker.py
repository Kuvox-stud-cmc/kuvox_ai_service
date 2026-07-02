"""Worker: consumes ingestion jobs from RabbitMQ."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import UTC, datetime
from uuid import uuid4

import aio_pika
from pydantic import ValidationError

from kuvox_ai.config import Settings, get_settings
from kuvox_ai.infrastructure.kuzu_client import KuzuClient
from kuvox_ai.infrastructure.object_storage_client import ObjectStorageClient
from kuvox_ai.infrastructure.rabbitmq_client import RabbitMQClient
from kuvox_ai.logging import configure_logging, get_logger
from kuvox_ai.modules.ingestion import IngestionFailed, IngestionRequested, IngestionService

logger = get_logger(__name__)


async def handle_message_body(
    body: bytes,
    *,
    service: IngestionService,
    rabbitmq: RabbitMQClient,
    completed_routing_key: str,
    failed_routing_key: str,
) -> None:
    try:
        request = IngestionRequested.model_validate_json(body)
    except ValidationError as exc:
        logger.warning("ingestion_worker.invalid_message", error=str(exc))
        return

    try:
        result = await service.ingest(request)
    except Exception as exc:  # noqa: BLE001
        failed = IngestionFailed(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            error_code=exc.__class__.__name__,
            error_message=str(exc),
        )
        await rabbitmq.publish_json(
            failed_routing_key,
            failed.model_dump(by_alias=True, mode="json"),
        )
        logger.warning("ingestion_worker.failed", media_id=request.media_id, error=str(exc))
        return

    await rabbitmq.publish_json(
        completed_routing_key,
        result.model_dump(by_alias=True, mode="json"),
    )
    logger.info("ingestion_worker.completed", media_id=request.media_id)


def build_service(
    settings: Settings,
    storage: ObjectStorageClient,
    kuzu: KuzuClient,
) -> IngestionService:
    return IngestionService(
        storage=storage,
        kuzu=kuzu,
        work_dir=settings.ingestion_work_dir,
    )


async def run_async() -> None:
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "ingestion_worker.starting",
        queue=settings.ingestion_requested_queue,
        routing_key=settings.ingestion_requested_routing_key,
    )

    storage = ObjectStorageClient.from_settings(settings)
    kuzu = KuzuClient.from_settings(settings)
    rabbitmq = RabbitMQClient.from_settings(settings)
    await storage.connect()
    await kuzu.connect()
    await rabbitmq.connect()

    service = build_service(settings, storage, kuzu)

    async def handle_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process(requeue=False):
            await handle_message_body(
                message.body,
                service=service,
                rabbitmq=rabbitmq,
                completed_routing_key=settings.ingestion_completed_routing_key,
                failed_routing_key=settings.ingestion_failed_routing_key,
            )

    await rabbitmq.consume_bound_queue(
        settings.ingestion_requested_queue,
        settings.ingestion_requested_routing_key,
        handle_message,
        prefetch_count=settings.ingestion_concurrency,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    logger.info("ingestion_worker.ready")
    try:
        await stop.wait()
    finally:
        logger.info("ingestion_worker.stopping")
        await rabbitmq.close()
        await kuzu.close()
        await storage.close()
        logger.info("ingestion_worker.stopped")


def run() -> None:
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
