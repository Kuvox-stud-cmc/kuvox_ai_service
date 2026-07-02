from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from kuvox_ai.modules.media_optimization import MediaOptimizationCompleted
from kuvox_ai.workers.media_optimization_worker import handle_message_body


def requested_body() -> bytes:
    return json.dumps(
        {
            "eventId": "evt-1",
            "eventType": "media.optimization.requested",
            "occurredAt": "2026-07-02T08:00:00Z",
            "mediaId": "media-1",
            "userId": "user-1",
            "bucketName": "kuvox-raw",
            "objectKey": "media/media-1/raw/original.mp4",
            "contentType": "video/mp4",
            "originalFileName": "demo.mp4",
            "sizeBytes": 123,
            "kind": "Video",
        }
    ).encode()


async def test_worker_publishes_completed_on_success(
    mock_rabbitmq: AsyncMock,
) -> None:
    service = AsyncMock()
    service.optimize.return_value = MediaOptimizationCompleted(
        event_id="evt-2",
        occurred_at=datetime(2026, 7, 2, tzinfo=UTC),
        source_event_id="evt-1",
        media_id="media-1",
        raw_bucket_name="kuvox-raw",
        raw_object_key="media/media-1/raw/original.mp4",
        raw_size_bytes=123,
    )

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="media.optimization.completed",
        failed_routing_key="media.optimization.failed",
    )

    mock_rabbitmq.publish_json.assert_awaited_once()
    assert mock_rabbitmq.publish_json.await_args.args[0] == "media.optimization.completed"
    assert mock_rabbitmq.publish_json.await_args.args[1]["sourceEventId"] == "evt-1"
    assert "projectId" not in mock_rabbitmq.publish_json.await_args.args[1]


async def test_worker_publishes_failed_on_service_error(
    mock_rabbitmq: AsyncMock,
) -> None:
    service = AsyncMock()
    service.optimize.side_effect = RuntimeError("nope")

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="media.optimization.completed",
        failed_routing_key="media.optimization.failed",
    )

    mock_rabbitmq.publish_json.assert_awaited_once()
    assert mock_rabbitmq.publish_json.await_args.args[0] == "media.optimization.failed"
    assert mock_rabbitmq.publish_json.await_args.args[1]["sourceEventId"] == "evt-1"
    assert mock_rabbitmq.publish_json.await_args.args[1]["errorCode"] == "RuntimeError"
    assert "projectId" not in mock_rabbitmq.publish_json.await_args.args[1]


async def test_worker_schedules_retry_before_terminal_failure(
    mock_rabbitmq: AsyncMock,
) -> None:
    service = AsyncMock()
    service.optimize.side_effect = RuntimeError("temporary")

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="media.optimization.completed",
        failed_routing_key="media.optimization.failed",
        queue_name="media.optimization.requested",
        retry_attempt=1,
        max_retry_attempts=3,
        headers={"x-kuvox-attempt": 1},
    )

    mock_rabbitmq.publish_retry.assert_awaited_once()
    assert mock_rabbitmq.publish_retry.await_args.args[:3] == (
        "media.optimization.requested",
        2,
        requested_body(),
    )
    mock_rabbitmq.publish_json.assert_not_awaited()
    mock_rabbitmq.publish_dlq.assert_not_awaited()


async def test_worker_final_failure_publishes_failed_and_dlq(
    mock_rabbitmq: AsyncMock,
) -> None:
    service = AsyncMock()
    service.optimize.side_effect = RuntimeError("permanent")

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="media.optimization.completed",
        failed_routing_key="media.optimization.failed",
        queue_name="media.optimization.requested",
        retry_attempt=3,
        max_retry_attempts=3,
        headers={"x-kuvox-attempt": 3},
    )

    mock_rabbitmq.publish_json.assert_awaited_once()
    assert mock_rabbitmq.publish_json.await_args.args[0] == "media.optimization.failed"
    mock_rabbitmq.publish_dlq.assert_awaited_once()
    mock_rabbitmq.publish_retry.assert_not_awaited()


async def test_worker_invalid_message_dlqs_when_queue_context_present(
    mock_rabbitmq: AsyncMock,
) -> None:
    service = AsyncMock()

    await handle_message_body(
        b'{"projectId":"project-1"}',
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="media.optimization.completed",
        failed_routing_key="media.optimization.failed",
        queue_name="media.optimization.requested",
        retry_attempt=0,
        max_retry_attempts=3,
    )

    service.optimize.assert_not_awaited()
    mock_rabbitmq.publish_dlq.assert_awaited_once()
    mock_rabbitmq.publish_json.assert_not_awaited()
