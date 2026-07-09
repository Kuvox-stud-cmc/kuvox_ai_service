"""FFprobe metadata helpers for ingestion."""

from __future__ import annotations

import asyncio
from pathlib import Path

from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.models import AudioMetadata, ImageMetadata, VideoMetadata
from kuvox_ai.modules.media_optimization.ffmpeg import FfmpegError, ffprobe_json
from kuvox_ai.modules.media_optimization.service import extract_basic_metadata, image_metadata

logger = get_logger(__name__)


async def probe_video_metadata(path: Path) -> VideoMetadata:
    try:
        return VideoMetadata.model_validate(extract_basic_metadata(await ffprobe_json(path)))
    except FfmpegError as exc:
        logger.warning(
            "ingestion.ffprobe_failed",
            path=str(path),
            error=str(exc),
        )
        return VideoMetadata()


async def probe_audio_metadata(path: Path) -> AudioMetadata:
    try:
        return AudioMetadata.model_validate(extract_basic_metadata(await ffprobe_json(path)))
    except FfmpegError as exc:
        logger.warning(
            "ingestion.audio_ffprobe_failed",
            path=str(path),
            error=str(exc),
        )
        return AudioMetadata()


async def probe_image_metadata(path: Path) -> ImageMetadata:
    try:
        return ImageMetadata.model_validate(await _image_metadata(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ingestion.image_metadata_failed",
            path=str(path),
            error=str(exc),
        )
        return ImageMetadata()


async def _image_metadata(path: Path) -> dict[str, object]:
    return await asyncio.to_thread(image_metadata, path)
