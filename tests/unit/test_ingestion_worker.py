from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from kuvox_ai.modules.ingestion import IngestionCompleted
from kuvox_ai.workers.ingestion_worker import handle_message_body


def requested_body() -> bytes:
    return json.dumps(
        {
            "eventId": "evt-1",
            "eventType": "ingestion.requested",
            "occurredAt": "2026-07-02T08:00:00Z",
            "mediaId": "media-1",
            "ownerId": "owner-1",
            "ownerKind": "User",
            "kind": "Video",
            "canonical": {
                "bucketName": "kuvox-canonical",
                "objectKey": "media/media-1/canonical.mp4",
                "contentType": "video/mp4",
                "sizeBytes": 456,
            },
        }
    ).encode()


async def test_worker_publishes_completed_on_success(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()
    service.ingest.return_value = IngestionCompleted(
        event_id="evt-2",
        occurred_at=datetime(2026, 7, 2, tzinfo=UTC),
        source_event_id="evt-1",
        media_id="media-1",
        shot_count=3,
    )

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="ingestion.completed",
        failed_routing_key="ingestion.failed",
    )

    mock_rabbitmq.publish_json.assert_awaited_once()
    assert mock_rabbitmq.publish_json.await_args.args[0] == "ingestion.completed"
    assert mock_rabbitmq.publish_json.await_args.args[1]["sourceEventId"] == "evt-1"
    assert mock_rabbitmq.publish_json.await_args.args[1]["shotCount"] == 3
    assert "projectId" not in mock_rabbitmq.publish_json.await_args.args[1]


async def test_worker_publishes_failed_on_service_error(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()
    service.ingest.side_effect = RuntimeError("nope")

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="ingestion.completed",
        failed_routing_key="ingestion.failed",
    )

    mock_rabbitmq.publish_json.assert_awaited_once()
    assert mock_rabbitmq.publish_json.await_args.args[0] == "ingestion.failed"
    assert mock_rabbitmq.publish_json.await_args.args[1]["sourceEventId"] == "evt-1"
    assert mock_rabbitmq.publish_json.await_args.args[1]["errorCode"] == "RuntimeError"
    assert "projectId" not in mock_rabbitmq.publish_json.await_args.args[1]


async def test_worker_acks_invalid_message_without_publish(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()

    await handle_message_body(
        b'{"projectId":"project-1"}',
        service=service,
        rabbitmq=mock_rabbitmq,
        completed_routing_key="ingestion.completed",
        failed_routing_key="ingestion.failed",
    )

    service.ingest.assert_not_awaited()
    mock_rabbitmq.publish_json.assert_not_awaited()
