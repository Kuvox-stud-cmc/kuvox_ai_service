from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kuvox_ai.modules.ingestion.models import IngestionRequested, MediaKind, OwnerKind, shot_id_for


def requested_payload() -> dict[str, object]:
    return {
        "eventId": "evt-1",
        "eventType": "ingestion.requested",
        "occurredAt": "2026-07-02T08:00:00Z",
        "mediaId": "media-1",
        "ownerId": "owner-1",
        "ownerKind": "Studio",
        "kind": "Video",
        "canonical": {
            "bucketName": "kuvox-canonical",
            "objectKey": "media/media-1/canonical.mp4",
            "contentType": "video/mp4",
            "sizeBytes": 456,
        },
        "proxy": {
            "bucketName": "kuvox-proxy",
            "objectKey": "media/media-1/proxy.mp4",
            "contentType": "video/mp4",
            "sizeBytes": 234,
        },
        "durationSeconds": 12.5,
        "width": 1920,
        "height": 1080,
        "frameRate": 29.97,
        "codec": "h265",
    }


def test_requested_event_parses_camel_case_and_has_no_project_id() -> None:
    event = IngestionRequested.model_validate_json(json.dumps(requested_payload()))

    assert event.media_id == "media-1"
    assert event.owner_kind == OwnerKind.studio
    assert event.kind == MediaKind.video
    assert event.canonical.object_key == "media/media-1/canonical.mp4"
    assert not hasattr(event, "project_id")


def test_requested_event_rejects_project_id() -> None:
    payload = requested_payload()
    payload["projectId"] = "project-1"

    with pytest.raises(ValidationError):
        IngestionRequested.model_validate(payload)


def test_shot_id_is_deterministic_and_zero_padded() -> None:
    assert shot_id_for("media-1", 7) == "media-1:shot:000007"
