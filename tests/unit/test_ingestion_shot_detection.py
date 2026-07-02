from __future__ import annotations

from pathlib import Path

from kuvox_ai.modules.ingestion.shot_detection import detect_video_shots, fallback_shot


def test_fallback_shot_covers_full_video() -> None:
    shot = fallback_shot("media-1", 42.25)

    assert shot.shot_id == "media-1:shot:000000"
    assert shot.start_seconds == 0
    assert shot.end_seconds == 42.25
    assert shot.duration_seconds == 42.25


async def test_shot_detection_falls_back_to_one_full_video_shot(tmp_path: Path) -> None:
    shots = await detect_video_shots(
        tmp_path / "missing.mp4",
        media_id="media-1",
        duration_seconds=9.5,
    )

    assert len(shots) == 1
    assert shots[0].shot_id == "media-1:shot:000000"
    assert shots[0].duration_seconds == 9.5
