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
from kuvox_ai.infrastructure.qdrant_client import QdrantClient
from kuvox_ai.infrastructure.rabbitmq_client import RabbitMQClient, retry_attempt
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
    queue_name: str | None = None,
    retry_attempt: int = 0,
    max_retry_attempts: int = 0,
    headers: dict[str, object] | None = None,
) -> None:
    try:
        request = IngestionRequested.model_validate_json(body)
    except ValidationError as exc:
        logger.warning("ingestion_worker.invalid_message", error=str(exc))
        if queue_name is not None:
            await rabbitmq.publish_dlq(
                queue_name,
                body,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                headers=headers,
            )
        return

    try:
        result = await service.ingest(request)
    except Exception as exc:  # noqa: BLE001
        if queue_name is not None and retry_attempt < max_retry_attempts:
            await rabbitmq.publish_retry(
                queue_name,
                retry_attempt + 1,
                body,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                headers=headers,
            )
            logger.warning(
                "ingestion_worker.retry_scheduled",
                media_id=request.media_id,
                attempt=retry_attempt + 1,
                error=str(exc),
            )
            return

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
        if queue_name is not None:
            await rabbitmq.publish_dlq(
                queue_name,
                body,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                headers=headers,
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
    qdrant: QdrantClient,
) -> IngestionService:
    return IngestionService(
        storage=storage,
        kuzu=kuzu,
        qdrant=qdrant,
        work_dir=settings.ingestion_work_dir,
        visual_collection_name=settings.visual_collection_name,
        visual_embedding_dim=settings.visual_embedding_dim,
        transcript_collection_name=settings.transcript_collection_name,
        audio_collection_name=settings.audio_collection_name,
        ocr_collection_name=settings.ocr_collection_name,
        text_embedding_model_name=settings.text_embedding_model_name,
        text_embedding_dim=settings.text_embedding_dim,
        text_embedding_device=settings.text_embedding_device,
        text_embedding_batch_size=settings.text_embedding_batch_size,
        whisper_model_name=settings.whisper_model_name,
        whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        audio_embedding_dim=settings.audio_embedding_dim,
        audio_embedding_device=settings.audio_embedding_device,
        audio_embedding_batch_size=settings.audio_embedding_batch_size,
        ocr_languages=settings.ocr_language_list,
        ocr_gpu=settings.ocr_gpu,
        ocr_min_confidence=settings.ocr_min_confidence,
        clip_model_name=settings.clip_model_name,
        clip_pretrained=settings.clip_pretrained,
        clip_device=settings.clip_device,
        clip_batch_size=settings.clip_batch_size,
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
    qdrant = QdrantClient.from_settings(settings)
    rabbitmq = RabbitMQClient.from_settings(settings)
    await storage.connect()
    await kuzu.connect()
    await qdrant.connect()
    await rabbitmq.connect()
    await rabbitmq.declare_retry_topology(
        settings.ingestion_requested_queue,
        settings.ingestion_requested_routing_key,
        settings.rabbitmq_retry_delay_list,
    )

    service = build_service(settings, storage, kuzu, qdrant)

    async def handle_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process(requeue=True):
            await handle_message_body(
                message.body,
                service=service,
                rabbitmq=rabbitmq,
                completed_routing_key=settings.ingestion_completed_routing_key,
                failed_routing_key=settings.ingestion_failed_routing_key,
                queue_name=settings.ingestion_requested_queue,
                retry_attempt=retry_attempt(dict(message.headers or {})),
                max_retry_attempts=settings.rabbitmq_retry_attempts,
                headers=dict(message.headers or {}),
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
        await qdrant.close()
        await kuzu.close()
        await storage.close()
        logger.info("ingestion_worker.stopped")


def run() -> None:
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
