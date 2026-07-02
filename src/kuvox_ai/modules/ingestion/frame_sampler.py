"""Representative frame sampling for detected video shots."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kuvox_ai.modules.ingestion.models import DetectedShot


@dataclass(frozen=True, slots=True)
class SampledFrame:
    shot: DetectedShot
    timestamp_seconds: float
    path: Path


class FrameSampler(Protocol):
    async def sample_frames(
        self,
        video_path: Path,
        shots: list[DetectedShot],
        output_dir: Path,
    ) -> list[SampledFrame]: ...


class FFmpegFrameSampler:
    async def sample_frames(
        self,
        video_path: Path,
        shots: list[DetectedShot],
        output_dir: Path,
    ) -> list[SampledFrame]:
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        frames: list[SampledFrame] = []
        for shot in shots:
            timestamp_seconds = midpoint_seconds(shot)
            output_path = output_dir / f"shot_{shot.shot_index:06d}.jpg"
            await extract_frame(video_path, timestamp_seconds, output_path)
            frames.append(
                SampledFrame(
                    shot=shot,
                    timestamp_seconds=timestamp_seconds,
                    path=output_path,
                )
            )
        return frames


def midpoint_seconds(shot: DetectedShot) -> float:
    return max(0.0, shot.start_seconds + (shot.duration_seconds / 2.0))


async def extract_frame(video_path: Path, timestamp_seconds: float, output_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp_seconds:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg frame extraction failed: {message}")
