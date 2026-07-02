from __future__ import annotations

from pathlib import Path

import pytest

import kuvox_ai.modules.ingestion.frame_sampler as frame_sampler_module
from kuvox_ai.modules.ingestion.frame_sampler import FFmpegFrameSampler, midpoint_seconds
from kuvox_ai.modules.ingestion.models import DetectedShot


def shot(index: int, start: float, end: float) -> DetectedShot:
    return DetectedShot(
        shot_id=f"media-1:shot:{index:06d}",
        media_id="media-1",
        shot_index=index,
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
    )


def test_midpoint_seconds_uses_shot_center() -> None:
    assert midpoint_seconds(shot(0, 10.0, 16.0)) == 13.0


async def test_frame_sampler_extracts_midpoint_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, float, Path]] = []

    async def fake_extract_frame(
        video_path: Path,
        timestamp_seconds: float,
        output_path: Path,
    ) -> None:
        calls.append((video_path, timestamp_seconds, output_path))

    monkeypatch.setattr(frame_sampler_module, "extract_frame", fake_extract_frame)

    sampler = FFmpegFrameSampler()
    video_path = tmp_path / "canonical.mp4"
    output_dir = tmp_path / "frames"
    frames = await sampler.sample_frames(
        video_path, [shot(0, 0.0, 12.0), shot(1, 20.0, 24.0)], output_dir
    )

    assert [frame.timestamp_seconds for frame in frames] == [6.0, 22.0]
    assert [frame.path.name for frame in frames] == ["shot_000000.jpg", "shot_000001.jpg"]
    assert calls == [
        (video_path, 6.0, output_dir / "shot_000000.jpg"),
        (video_path, 22.0, output_dir / "shot_000001.jpg"),
    ]


async def test_extract_frame_calls_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command: tuple[str, ...] | None = None

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_create_subprocess_exec(*args: str, **_: object) -> Process:
        nonlocal command
        command = args
        return Process()

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.frame_sampler.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    output_path = tmp_path / "frame.jpg"
    await frame_sampler_module.extract_frame(tmp_path / "video.mp4", 3.25, output_path)

    assert command == (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "3.250000",
        "-i",
        str(tmp_path / "video.mp4"),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    )
