"""Video shot detection with a one-shot fallback."""

from __future__ import annotations

import asyncio
from pathlib import Path

from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.models import DetectedShot, shot_id_for

logger = get_logger(__name__)


async def detect_video_shots(
    path: Path,
    *,
    media_id: str,
    duration_seconds: float,
) -> list[DetectedShot]:
    return await asyncio.to_thread(_detect_video_shots_sync, path, media_id, duration_seconds)


def fallback_shot(media_id: str, duration_seconds: float) -> DetectedShot:
    end_seconds = max(0.0, duration_seconds)
    return DetectedShot(
        shot_id=shot_id_for(media_id, 0),
        media_id=media_id,
        shot_index=0,
        start_seconds=0.0,
        end_seconds=end_seconds,
        duration_seconds=end_seconds,
    )


def _detect_video_shots_sync(
    path: Path, media_id: str, duration_seconds: float
) -> list[DetectedShot]:
    try:
        from scenedetect import ContentDetector, detect  # type: ignore[import-not-found]

        scenes = detect(str(path), ContentDetector())
        shots = [
            DetectedShot(
                shot_id=shot_id_for(media_id, index),
                media_id=media_id,
                shot_index=index,
                start_seconds=float(start.get_seconds()),
                end_seconds=float(end.get_seconds()),
                duration_seconds=max(0.0, float(end.get_seconds()) - float(start.get_seconds())),
            )
            for index, (start, end) in enumerate(scenes)
        ]
        if shots:
            return shots
    except Exception as exc:  # noqa: BLE001
        logger.warning("ingestion.shot_detection_fallback", error=str(exc))

    return [fallback_shot(media_id, duration_seconds)]
