"""Ingestion service MVP 1 orchestration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import kuvox_ai.modules.ingestion.service as service_module
from kuvox_ai.modules.ingestion import IngestionService
from kuvox_ai.modules.ingestion.models import DetectedShot, IngestionRequested, VideoMetadata


def requested_payload() -> dict[str, object]:
    return {
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
        "durationSeconds": 12.0,
        "width": 1920,
        "height": 1080,
        "frameRate": 30.0,
        "codec": "h265",
    }


async def test_ingest_downloads_canonical_detects_shots_and_writes_graph(
    mock_kuzu: AsyncMock,
    mock_qdrant: AsyncMock,
    mock_storage: AsyncMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IngestionRequested.model_validate(requested_payload())
    writer = AsyncMock()
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=12,
        duration_seconds=12,
    )

    probe = AsyncMock(return_value=VideoMetadata(duration_seconds=12, width=1280, height=720))
    detect = AsyncMock(return_value=[shot])
    monkeypatch.setattr(service_module, "probe_video_metadata", probe)
    monkeypatch.setattr(service_module, "detect_video_shots", detect)

    svc = IngestionService(
        kuzu=mock_kuzu,
        qdrant=mock_qdrant,
        storage=mock_storage,
        work_dir=tmp_path,
        writer=writer,
    )

    result = await svc.ingest(request)

    mock_storage.download_file.assert_awaited_once()
    assert mock_storage.download_file.await_args.args[:2] == (
        "kuvox-canonical",
        "media/media-1/canonical.mp4",
    )
    assert mock_storage.download_file.await_args.args[2].name == "canonical.mp4"
    detect.assert_awaited_once()
    writer.write_video_with_shots.assert_awaited_once()
    assert result.source_event_id == "evt-1"
    assert result.media_id == "media-1"
    assert result.shot_count == 1
