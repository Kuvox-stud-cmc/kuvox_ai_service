from __future__ import annotations

from pathlib import Path

import pytest

from kuvox_ai.modules.ingestion.metadata import probe_video_metadata
from kuvox_ai.modules.media_optimization import ffmpeg


async def test_probe_video_metadata_returns_empty_metadata_when_ffprobe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_ffprobe_json(path: Path) -> dict[str, object]:
        raise ffmpeg.FfmpegError("ffprobe missing")

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.metadata.ffprobe_json",
        fake_ffprobe_json,
    )

    metadata = await probe_video_metadata(tmp_path / "video.mp4")

    assert metadata.duration_seconds is None
    assert metadata.width is None
    assert metadata.height is None
