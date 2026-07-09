"""Rendering RabbitMQ handlers registered by the FastAPI app lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from kuvox_ai.infrastructure.rabbitmq_client import RabbitMQClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.rendering.models import (
    RenderingCompleted,
    RenderingFailed,
    RenderingStarted,
    RenderJob,
)
from kuvox_ai.modules.rendering.service import RenderingService

logger = get_logger(__name__)


async def handle_message_body(
    body: bytes,
    *,
    service: RenderingService,
    rabbitmq: RabbitMQClient,
    started_routing_key: str,
    completed_routing_key: str,
    failed_routing_key: str,
    queue_name: str | None = None,
    retry_attempt: int = 0,
    max_retry_attempts: int = 0,
    headers: dict[str, object] | None = None,
) -> None:
    try:
        request = RenderJob.model_validate_json(body)
    except ValidationError as exc:
        logger.warning("rendering_worker.invalid_message", error=str(exc))
        if queue_name is not None:
            await rabbitmq.publish_dlq(
                queue_name,
                body,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                headers=headers,
            )
        return

    started_at = datetime.now(UTC)
    started = RenderingStarted(
        event_id=str(uuid4()),
        occurred_at=started_at,
        source_event_id=request.event_id,
        render_job_id=request.render_job_id,
        started_at=started_at,
    )
    await rabbitmq.publish_json(
        started_routing_key,
        started.model_dump(by_alias=True, mode="json"),
    )

    try:
        result = await service.render(request)
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
                "rendering_worker.retry_scheduled",
                render_job_id=request.render_job_id,
                attempt=retry_attempt + 1,
                error=str(exc),
            )
            return

        finished_at = datetime.now(UTC)
        failed = RenderingFailed(
            event_id=str(uuid4()),
            occurred_at=finished_at,
            source_event_id=request.event_id,
            render_job_id=request.render_job_id,
            error_code=exc.__class__.__name__,
            error_message=str(exc) or exc.__class__.__name__,
            finished_at=finished_at,
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
            "rendering_worker.failed",
            render_job_id=request.render_job_id,
            error=str(exc),
        )
        return

    finished_at = datetime.now(UTC)
    completed = RenderingCompleted(
        event_id=str(uuid4()),
        occurred_at=finished_at,
        source_event_id=request.event_id,
        render_job_id=request.render_job_id,
        output_bucket_name=result.output_bucket_name or request.output_bucket_name,
        output_storage_key=result.output_storage_key,
        output_content_type=result.output_content_type or request.output_content_type,
        output_size_bytes=result.output_size_bytes,
        finished_at=finished_at,
    )
    await rabbitmq.publish_json(
        completed_routing_key,
        completed.model_dump(by_alias=True, mode="json"),
    )
    logger.info("rendering_worker.completed", render_job_id=request.render_job_id)
