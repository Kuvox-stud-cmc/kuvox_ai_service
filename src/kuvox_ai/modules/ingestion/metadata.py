"""FFprobe metadata helpers for ingestion."""

from __future__ import annotations

from pathlib import Path

from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.models import VideoMetadata
from kuvox_ai.modules.media_optimization.ffmpeg import FfmpegError, ffprobe_json
from kuvox_ai.modules.media_optimization.service import extract_basic_metadata

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
