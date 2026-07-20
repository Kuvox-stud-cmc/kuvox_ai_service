"""Audio extraction helpers for shot-level ingestion indexing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.media_optimization import ffmpeg

logger = get_logger(__name__)

FULL_AUDIO_SAMPLE_RATE = 16_000
SHOT_AUDIO_SAMPLE_RATE = 48_000
AUDIO_CLIP_TIMESTAMP_PRECISION = 6
FULL_AUDIO_EXTRACTION_CONTRACT_ID = "ffmpeg-full-input-v1;mono;sample-rate=16000;codec=pcm_s16le"
SHOT_AUDIO_EXTRACTION_CONTRACT_ID = (
    "ffmpeg-seek-before-input-v1;start=6dp;duration=6dp;mono;sample-rate=48000;codec=pcm_s16le"
)
AUDIO_CLIP_BOUNDARY_CONTRACT_ID = "start=shot-start;end=shot-end;duration=max-zero-shot-duration-v1"


@dataclass(frozen=True, slots=True)
class ShotAudioClip:
    shot: DetectedShot
    path: Path
    clip_start_seconds: float
    clip_end_seconds: float
    clip_duration_seconds: float


class AudioExtractor(Protocol):
    async def has_audio_stream(self, video_path: Path) -> bool: ...

    async def extract_full_audio(self, video_path: Path, output_dir: Path) -> Path: ...

    async def extract_shot_audio_clips(
        self,
        video_path: Path,
        shots: list[DetectedShot],
        output_dir: Path,
    ) -> list[ShotAudioClip]: ...


class FFmpegAudioExtractor:
    async def has_audio_stream(self, video_path: Path) -> bool:
        try:
            metadata = await ffmpeg.ffprobe_json(video_path)
        except ffmpeg.FfmpegError as exc:
            logger.warning(
                "ingestion.audio_probe_failed",
                path=str(video_path),
                error=str(exc),
            )
            return False

        streams = metadata.get("streams")
        if not isinstance(streams, list):
            return False
        return any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
        )

    async def extract_full_audio(self, video_path: Path, output_dir: Path) -> Path:
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        output_path = output_dir / "full_audio.wav"
        await ffmpeg.run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(FULL_AUDIO_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )
        return output_path

    async def extract_shot_audio_clips(
        self,
        video_path: Path,
        shots: list[DetectedShot],
        output_dir: Path,
    ) -> list[ShotAudioClip]:
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        clips: list[ShotAudioClip] = []
        for shot in shots:
            output_path = output_dir / f"shot_{shot.shot_index:06d}.wav"
            clip_duration_seconds = max(0.0, shot.duration_seconds)
            await ffmpeg.run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{shot.start_seconds:.{AUDIO_CLIP_TIMESTAMP_PRECISION}f}",
                    "-i",
                    str(video_path),
                    "-t",
                    f"{clip_duration_seconds:.{AUDIO_CLIP_TIMESTAMP_PRECISION}f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SHOT_AUDIO_SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ]
            )
            clips.append(
                ShotAudioClip(
                    shot=shot,
                    path=output_path,
                    clip_start_seconds=shot.start_seconds,
                    clip_end_seconds=shot.end_seconds,
                    clip_duration_seconds=clip_duration_seconds,
                )
            )
        return clips
