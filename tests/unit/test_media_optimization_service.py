from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kuvox_ai.modules.media_optimization import ffmpeg
from kuvox_ai.modules.media_optimization import service as media_optimization_service
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


async def fake_write_webp_image_variant(
    _source: Path,
    destination: Path,
    *,
    max_width: int,
    quality: int,
) -> None:
    del max_width, quality
    await asyncio.to_thread(destination.write_bytes, b"optimized")


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
                ("kuvox-thumbnails", "media/media-1/poster.png", "image/png"),
            ],
        ),
        (
            "Audio",
            "demo.wav",
            "audio/wav",
            [
                ("kuvox-canonical", "media/media-1/canonical.opus", "audio/opus"),
                ("kuvox-thumbnails", "media/media-1/waveform.png", "image/png"),
            ],
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
    monkeypatch.setattr(
        media_optimization_service,
        "write_webp_image_variant",
        fake_write_webp_image_variant,
    )
    monkeypatch.setattr(
        media_optimization_service,
        "image_metadata",
        lambda _path: {"width": 320, "height": 180},
    )

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


@pytest.mark.parametrize(
    ("kind", "filename", "content_type"),
    [
        ("Video", "demo.mp4", "video/mp4"),
        ("Audio", "demo.wav", "audio/wav"),
    ],
)
async def test_probe_failure_does_not_block_canonical_completion(
    kind: str,
    filename: str,
    content_type: str,
    tmp_path: Path,
    mock_storage: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_ffprobe(_path: Path) -> dict[str, Any]:
        raise ffmpeg.FfmpegError("ffprobe missing")

    mock_storage.download_file.side_effect = fake_download_file
    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
    monkeypatch.setattr(ffmpeg, "ffprobe_json", fail_ffprobe)

    result = await make_service(mock_storage, tmp_path).optimize(
        make_request(kind, filename=filename, content_type=content_type)
    )

    assert result.canonical is not None
    assert result.duration_seconds is None


@pytest.mark.parametrize(
    ("kind", "filename", "content_type", "canonical_name", "expected_upload"),
    [
        (
            "Video",
            "demo.mp4",
            "video/mp4",
            "canonical.mp4",
            ("kuvox-canonical", "media/media-1/canonical.mp4", "video/mp4"),
        ),
        (
            "Audio",
            "demo.wav",
            "audio/wav",
            "canonical.opus",
            ("kuvox-canonical", "media/media-1/canonical.opus", "audio/opus"),
        ),
    ],
)
async def test_optional_preview_failure_does_not_block_canonical_completion(
    kind: str,
    filename: str,
    content_type: str,
    canonical_name: str,
    expected_upload: tuple[str, str, str],
    tmp_path: Path,
    mock_storage: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_optional_outputs(args: list[str], timeout_seconds: int = 900) -> None:
        del timeout_seconds
        output_path = Path(args[-1])
        if output_path.name != canonical_name:
            raise ffmpeg.FfmpegError("optional preview failed")
        await asyncio.to_thread(output_path.write_bytes, b"optimized")

    mock_storage.download_file.side_effect = fake_download_file
    monkeypatch.setattr(ffmpeg, "run_command", fail_optional_outputs)
    monkeypatch.setattr(ffmpeg, "ffprobe_json", fake_ffprobe_json)

    result = await make_service(mock_storage, tmp_path).optimize(
        make_request(kind, filename=filename, content_type=content_type)
    )

    actual_uploads = [
        (call.args[1], call.args[2], call.kwargs["content_type"])
        for call in mock_storage.upload_file.call_args_list
    ]
    assert actual_uploads == [expected_upload]
    assert result.canonical is not None
    assert result.proxy is None
    assert result.thumbnail is None


async def test_image_optimization_does_not_require_ffmpeg_webp_encoder(
    tmp_path: Path,
    mock_storage: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def download_image(_bucket: str, _key: str, destination: Path) -> None:
        def write_image() -> None:
            from PIL import Image

            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (32, 24), (255, 0, 0, 128)).save(destination)

        await asyncio.to_thread(write_image)

    async def fail_if_ffmpeg_is_used(args: list[str], timeout_seconds: int = 900) -> None:
        del args, timeout_seconds
        raise AssertionError("Image optimization should use Pillow, not FFmpeg.")

    mock_storage.download_file.side_effect = download_image
    monkeypatch.setattr(ffmpeg, "run_command", fail_if_ffmpeg_is_used)

    result = await make_service(mock_storage, tmp_path).optimize(
        make_request("Image", filename="demo.png", content_type="image/png")
    )

    actual_uploads = [
        (call.args[1], call.args[2], call.kwargs["content_type"])
        for call in mock_storage.upload_file.call_args_list
    ]
    assert actual_uploads == [
        ("kuvox-canonical", "media/media-1/canonical.webp", "image/webp"),
        ("kuvox-thumbnails", "media/media-1/thumb.webp", "image/webp"),
    ]
    assert result.width == 32
    assert result.height == 24
