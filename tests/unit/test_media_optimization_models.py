from __future__ import annotations

import json
from datetime import UTC, datetime

from kuvox_ai.modules.media_optimization import (
    MediaKind,
    MediaOptimizationCompleted,
    MediaOptimizationFailed,
    MediaOptimizationRequested,
    OptimizedObject,
)


def requested_payload(kind: str = "Video") -> dict[str, object]:
    return {
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
        "kind": kind,
    }


def test_requested_event_parses_camel_case_and_enum_string() -> None:
    event = MediaOptimizationRequested.model_validate_json(json.dumps(requested_payload("Video")))

    assert event.event_id == "evt-1"
    assert event.kind == MediaKind.video


def test_completed_event_dumps_camel_case() -> None:
    event = MediaOptimizationCompleted(
        event_id="evt-2",
        occurred_at=datetime(2026, 7, 2, tzinfo=UTC),
        source_event_id="evt-1",
        media_id="media-1",
        canonical=OptimizedObject(
            bucket_name="kuvox-canonical",
            object_key="media/media-1/canonical.mp4",
            content_type="video/mp4",
            size_bytes=456,
        ),
        raw_bucket_name="kuvox-raw",
        raw_object_key="media/media-1/raw/original.mp4",
        raw_size_bytes=123,
    )

    payload = event.model_dump(by_alias=True, mode="json")

    assert payload["eventId"] == "evt-2"
    assert payload["eventType"] == "media.optimization.completed"
    assert payload["sourceEventId"] == "evt-1"
    assert "projectId" not in payload
    assert payload["rawObjectKey"] == "media/media-1/raw/original.mp4"


def test_failed_event_dumps_camel_case() -> None:
    event = MediaOptimizationFailed(
        event_id="evt-3",
        occurred_at=datetime(2026, 7, 2, tzinfo=UTC),
        source_event_id="evt-1",
        media_id="media-1",
        error_code="FfmpegError",
        error_message="failed",
    )

    payload = event.model_dump(by_alias=True, mode="json")

    assert payload["eventType"] == "media.optimization.failed"
    assert payload["sourceEventId"] == "evt-1"
    assert "projectId" not in payload
    assert payload["errorCode"] == "FfmpegError"
