from __future__ import annotations

from pathlib import Path

import pytest

from kuvox_ai.modules.ingestion.audio import FFmpegAudioExtractor
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.media_optimization import ffmpeg


def shot() -> DetectedShot:
    return DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=2.0,
        end_seconds=5.5,
        duration_seconds=3.5,
    )


async def test_has_audio_stream_detects_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_ffprobe_json(path: Path) -> dict[str, object]:
        return {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]}

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.audio.ffmpeg.ffprobe_json",
        fake_ffprobe_json,
    )

    assert await FFmpegAudioExtractor().has_audio_stream(tmp_path / "video.mp4") is True


async def test_has_audio_stream_returns_false_without_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_ffprobe_json(path: Path) -> dict[str, object]:
        return {"streams": [{"codec_type": "video"}]}

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.audio.ffmpeg.ffprobe_json",
        fake_ffprobe_json,
    )

    assert await FFmpegAudioExtractor().has_audio_stream(tmp_path / "video.mp4") is False


async def test_has_audio_stream_returns_false_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_ffprobe_json(path: Path) -> dict[str, object]:
        raise ffmpeg.FfmpegError("ffprobe missing")

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.audio.ffmpeg.ffprobe_json",
        fake_ffprobe_json,
    )

    assert await FFmpegAudioExtractor().has_audio_stream(tmp_path / "video.mp4") is False


async def test_extract_full_audio_calls_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    async def fake_run_command(args: list[str], timeout_seconds: int = 900) -> None:
        calls.append(args)

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.audio.ffmpeg.run_command",
        fake_run_command,
    )

    output = await FFmpegAudioExtractor().extract_full_audio(
        tmp_path / "video.mp4",
        tmp_path / "audio",
    )

    assert output == tmp_path / "audio" / "full_audio.wav"
    assert calls[0] == [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_path / "video.mp4"),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


async def test_extract_shot_audio_clips_calls_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    async def fake_run_command(args: list[str], timeout_seconds: int = 900) -> None:
        calls.append(args)

    monkeypatch.setattr(
        "kuvox_ai.modules.ingestion.audio.ffmpeg.run_command",
        fake_run_command,
    )

    clips = await FFmpegAudioExtractor().extract_shot_audio_clips(
        tmp_path / "video.mp4",
        [shot()],
        tmp_path / "clips",
    )

    assert clips[0].path == tmp_path / "clips" / "shot_000000.wav"
    assert clips[0].clip_start_seconds == 2.0
    assert calls[0] == [
        "ffmpeg",
        "-y",
        "-ss",
        "2.000000",
        "-i",
        str(tmp_path / "video.mp4"),
        "-t",
        "3.500000",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        str(clips[0].path),
    ]
