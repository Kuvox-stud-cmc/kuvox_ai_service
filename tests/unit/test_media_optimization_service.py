from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kuvox_ai.modules.media_optimization import ffmpeg
from kuvox_ai.modules.media_optimization.models import MediaOptimizationRequested
from kuvox_ai.modules.media_optimization.service import MediaOptimizationService


def make_request(kind: str, *, filename: str, content_type: str) -> MediaOptimizationRequested:
    payload = {
        "eventId": "evt-1",
        "eventType": "media.optimization.requested",
        "occurredAt": "2026-07-02T08:00:00Z",
        "mediaId": "media-1",
        "userId": "user-1",
        "bucketName": "kuvox-raw",
        "objectKey": f"media/media-1/raw/{filename}",
        "contentType": content_type,
        "originalFileName": filename,
        "sizeBytes": 123,
        "kind": kind,
    }
    return MediaOptimizationRequested.model_validate_json(json.dumps(payload))


def make_service(storage: AsyncMock, work_dir: Path) -> MediaOptimizationService:
    return MediaOptimizationService(
        storage=storage,
        canonical_bucket="kuvox-canonical",
        proxy_bucket="kuvox-proxy",
        thumbnail_bucket="kuvox-thumbnails",
        work_dir=work_dir,
    )


async def fake_download_file(_bucket: str, _key: str, destination: Path) -> None:
    def write_file() -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"raw")

    await asyncio.to_thread(write_file)


async def fake_run_command(args: list[str], timeout_seconds: int = 900) -> None:
    del timeout_seconds
    await asyncio.to_thread(Path(args[-1]).write_bytes, b"optimized")


async def fake_ffprobe_json(_path: Path) -> dict[str, Any]:
    return {
        "format": {"duration": "10.5"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "codec_name": "hevc",
            }
        ],
    }


@pytest.mark.parametrize(
    ("kind", "filename", "content_type", "expected_uploads"),
    [
        (
            "Video",
            "demo.mp4",
            "video/mp4",
            [
                ("kuvox-canonical", "media/media-1/canonical.mp4", "video/mp4"),
                ("kuvox-proxy", "media/media-1/proxy.mp4", "video/mp4"),
                ("kuvox-thumbnails", "media/media-1/poster.webp", "image/webp"),
            ],
        ),
        (
            "Audio",
            "demo.wav",
            "audio/wav",
            [("kuvox-canonical", "media/media-1/canonical.opus", "audio/opus")],
        ),
        (
            "Image",
            "demo.png",
            "image/png",
            [
                (
                    "kuvox-canonical",
                    "media/media-1/canonical.webp",
                    "image/webp",
                ),
                ("kuvox-thumbnails", "media/media-1/thumb.webp", "image/webp"),
            ],
        ),
    ],
)
async def test_service_uploads_deterministic_outputs(
    kind: str,
    filename: str,
    content_type: str,
    expected_uploads: list[tuple[str, str, str]],
    tmp_path: Path,
    mock_storage: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_storage.download_file.side_effect = fake_download_file
    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
    monkeypatch.setattr(ffmpeg, "ffprobe_json", fake_ffprobe_json)

    result = await make_service(mock_storage, tmp_path).optimize(
        make_request(kind, filename=filename, content_type=content_type)
    )

    actual_uploads = [
        (call.args[1], call.args[2], call.kwargs["content_type"])
        for call in mock_storage.upload_file.call_args_list
    ]
    assert actual_uploads == expected_uploads
    assert result.raw_object_key == f"media/media-1/raw/{filename}"


async def test_ffmpeg_failure_cleans_temp_files(
    tmp_path: Path,
    mock_storage: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_run_command(args: list[str], timeout_seconds: int = 900) -> None:
        del args, timeout_seconds
        raise ffmpeg.FfmpegError("boom")

    mock_storage.download_file.side_effect = fake_download_file
    monkeypatch.setattr(ffmpeg, "run_command", fail_run_command)

    with pytest.raises(ffmpeg.FfmpegError):
        await make_service(mock_storage, tmp_path).optimize(
            make_request("Video", filename="demo.mp4", content_type="video/mp4")
        )
