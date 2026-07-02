"""FFprobe metadata helpers for ingestion."""

from __future__ import annotations

from pathlib import Path

from kuvox_ai.modules.ingestion.models import VideoMetadata
from kuvox_ai.modules.media_optimization.ffmpeg import ffprobe_json
from kuvox_ai.modules.media_optimization.service import extract_basic_metadata


async def probe_video_metadata(path: Path) -> VideoMetadata:
    return VideoMetadata.model_validate(extract_basic_metadata(await ffprobe_json(path)))
