"""Audio extraction helpers for shot-level ingestion indexing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.media_optimization import ffmpeg


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
        metadata = await ffmpeg.ffprobe_json(video_path)
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
                "16000",
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
                    f"{shot.start_seconds:.6f}",
                    "-i",
                    str(video_path),
                    "-t",
                    f"{clip_duration_seconds:.6f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
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
