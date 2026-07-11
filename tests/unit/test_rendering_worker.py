from __future__ import annotations

import json
from unittest.mock import AsyncMock

from kuvox_ai.modules.rendering.models import RenderResult
from kuvox_ai.workers.rendering_worker import handle_message_body


def requested_body() -> bytes:
    return json.dumps(
        {
            "eventId": "evt-1",
            "eventType": "rendering.requested",
            "occurredAt": "2026-07-09T00:00:00Z",
            "renderJobId": "job-1",
            "timelineId": "timeline-1",
            "projectId": "project-1",
            "revisionId": "revision-1",
            "revisionNumber": 7,
            "requestedByUserId": "user-1",
            "settings": {"format": "mp4", "width": 1920, "height": 1080},
            "documentJson": {"media": {}, "tracks": [], "snapshotMarker": "revision-7"},
            "mediaSources": [],
            "outputBucketName": "kuvox-renders",
            "outputStorageKey": "renders/job-1.mp4",
            "outputContentType": "video/mp4",
        }
    ).encode()


async def test_worker_publishes_started_and_completed_on_success(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()
    service.render.return_value = RenderResult(
        render_job_id="job-1",
        output_storage_key="renders/job-1.mp4",
        output_size_bytes=123,
    )

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        started_routing_key="rendering.started",
        completed_routing_key="rendering.completed",
        failed_routing_key="rendering.failed",
    )

    assert mock_rabbitmq.publish_json.await_count == 2
    assert mock_rabbitmq.publish_json.await_args_list[0].args[0] == "rendering.started"
    assert mock_rabbitmq.publish_json.await_args_list[0].args[1]["renderJobId"] == "job-1"
    assert mock_rabbitmq.publish_json.await_args_list[1].args[0] == "rendering.completed"
    assert mock_rabbitmq.publish_json.await_args_list[1].args[1]["sourceEventId"] == "evt-1"
    assert mock_rabbitmq.publish_json.await_args_list[1].args[1]["outputSizeBytes"] == 123
    rendered_job = service.render.await_args.args[0]
    assert rendered_job.revision_number == 7
    assert rendered_job.document_json["snapshotMarker"] == "revision-7"


async def test_worker_schedules_retry_before_terminal_failure(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()
    service.render.side_effect = RuntimeError("temporary")

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        started_routing_key="rendering.started",
        completed_routing_key="rendering.completed",
        failed_routing_key="rendering.failed",
        queue_name="kuvox.rendering",
        retry_attempt=1,
        max_retry_attempts=3,
        headers={"x-kuvox-attempt": 1},
    )

    mock_rabbitmq.publish_retry.assert_awaited_once()
    assert mock_rabbitmq.publish_retry.await_args.args[:3] == (
        "kuvox.rendering",
        2,
        requested_body(),
    )
    assert mock_rabbitmq.publish_json.await_count == 1
    assert mock_rabbitmq.publish_json.await_args.args[0] == "rendering.started"
    mock_rabbitmq.publish_dlq.assert_not_awaited()


async def test_worker_final_failure_publishes_failed_and_dlq(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()
    service.render.side_effect = RuntimeError("permanent")

    await handle_message_body(
        requested_body(),
        service=service,
        rabbitmq=mock_rabbitmq,
        started_routing_key="rendering.started",
        completed_routing_key="rendering.completed",
        failed_routing_key="rendering.failed",
        queue_name="kuvox.rendering",
        retry_attempt=3,
        max_retry_attempts=3,
        headers={"x-kuvox-attempt": 3},
    )

    assert mock_rabbitmq.publish_json.await_count == 2
    assert mock_rabbitmq.publish_json.await_args_list[1].args[0] == "rendering.failed"
    assert mock_rabbitmq.publish_json.await_args_list[1].args[1]["errorCode"] == "RuntimeError"
    mock_rabbitmq.publish_dlq.assert_awaited_once()
    mock_rabbitmq.publish_retry.assert_not_awaited()


async def test_worker_invalid_message_dlqs_and_does_not_render(mock_rabbitmq: AsyncMock) -> None:
    service = AsyncMock()

    await handle_message_body(
        b'{"projectId":"project-1"}',
        service=service,
        rabbitmq=mock_rabbitmq,
        started_routing_key="rendering.started",
        completed_routing_key="rendering.completed",
        failed_routing_key="rendering.failed",
        queue_name="kuvox.rendering",
        retry_attempt=0,
        max_retry_attempts=3,
    )

    service.render.assert_not_awaited()
    mock_rabbitmq.publish_dlq.assert_awaited_once()
    mock_rabbitmq.publish_json.assert_not_awaited()
