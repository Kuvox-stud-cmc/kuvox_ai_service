"""Worker: consumes media optimization jobs from RabbitMQ."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import UTC, datetime
from uuid import uuid4

import aio_pika
from pydantic import ValidationError

from kuvox_ai.config import Settings, get_settings
from kuvox_ai.infrastructure.object_storage_client import ObjectStorageClient
from kuvox_ai.infrastructure.rabbitmq_client import RabbitMQClient, retry_attempt
from kuvox_ai.logging import configure_logging, get_logger
from kuvox_ai.modules.media_optimization import (
    MediaOptimizationFailed,
    MediaOptimizationRequested,
    MediaOptimizationService,
    ffmpeg,
)

logger = get_logger(__name__)


async def handle_message_body(
    body: bytes,
    *,
    service: MediaOptimizationService,
    rabbitmq: RabbitMQClient,
    completed_routing_key: str,
    failed_routing_key: str,
    queue_name: str | None = None,
    retry_attempt: int = 0,
    max_retry_attempts: int = 0,
    headers: dict[str, object] | None = None,
) -> None:
    try:
        request = MediaOptimizationRequested.model_validate_json(body)
    except ValidationError as exc:
        logger.warning("media_optimization_worker.invalid_message", error=str(exc))
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
        result = await service.optimize(request)
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
                "media_optimization_worker.retry_scheduled",
                media_id=request.media_id,
                attempt=retry_attempt + 1,
                error=str(exc),
            )
            return

        failed = MediaOptimizationFailed(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            error_code=exc.__class__.__name__,
            error_message=str(exc) or exc.__class__.__name__,
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
        logger.warning(
            "media_optimization_worker.failed",
            media_id=request.media_id,
            error=str(exc),
        )
        return

    await rabbitmq.publish_json(
        completed_routing_key,
        result.model_dump(by_alias=True, mode="json"),
    )
    logger.info("media_optimization_worker.completed", media_id=request.media_id)


def build_service(settings: Settings, storage: ObjectStorageClient) -> MediaOptimizationService:
    return MediaOptimizationService(
        storage=storage,
        canonical_bucket=settings.s3_canonical_bucket,
        proxy_bucket=settings.s3_proxy_bucket,
        thumbnail_bucket=settings.s3_thumbnail_bucket,
        work_dir=settings.media_work_dir,
        video_canonical_crf=settings.video_canonical_crf,
        video_proxy_crf=settings.video_proxy_crf,
        video_proxy_max_width=settings.video_proxy_max_width,
        image_max_width=settings.image_max_width,
        thumbnail_width=settings.thumbnail_width,
    )


async def run_async() -> None:
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "media_optimization_worker.starting",
        queue=settings.media_optimization_requested_queue,
        routing_key=settings.media_optimization_requested_routing_key,
    )
    try:
        logger.info(
            "media_optimization_worker.ffmpeg_resolved",
            ffmpeg=ffmpeg.resolve_ffmpeg_exe(),
            ffprobe=ffmpeg.resolve_ffprobe_exe(),
            work_dir=str(settings.media_work_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("media_optimization_worker.ffmpeg_resolution_failed", error=str(exc))

    storage = ObjectStorageClient.from_settings(settings)
    rabbitmq = RabbitMQClient.from_settings(settings)
    await storage.connect()
    await rabbitmq.connect()
    await rabbitmq.declare_retry_topology(
        settings.media_optimization_requested_queue,
        settings.media_optimization_requested_routing_key,
        settings.rabbitmq_retry_delay_list,
    )

    service = build_service(settings, storage)

    async def handle_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process(requeue=True):
            await handle_message_body(
                message.body,
                service=service,
                rabbitmq=rabbitmq,
                completed_routing_key=settings.media_optimization_completed_routing_key,
                failed_routing_key=settings.media_optimization_failed_routing_key,
                queue_name=settings.media_optimization_requested_queue,
                retry_attempt=retry_attempt(dict(message.headers or {})),
                max_retry_attempts=settings.rabbitmq_retry_attempts,
                headers=dict(message.headers or {}),
            )

    await rabbitmq.consume_bound_queue(
        settings.media_optimization_requested_queue,
        settings.media_optimization_requested_routing_key,
        handle_message,
        prefetch_count=settings.media_optimization_concurrency,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    logger.info("media_optimization_worker.ready")
    try:
        await stop.wait()
    finally:
        logger.info("media_optimization_worker.stopping")
        await rabbitmq.close()
        await storage.close()
        logger.info("media_optimization_worker.stopped")


def run() -> None:
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
