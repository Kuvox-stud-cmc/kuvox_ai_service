from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kuvox_ai.modules.ingestion.transcript as transcript_module
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.transcript import (
    FasterWhisperTranscriber,
    TranscriptSegment,
    align_transcript_to_shots,
    resolve_whisper_compute_type,
)


def shot(index: int, start: float, end: float) -> DetectedShot:
    return DetectedShot(
        shot_id=f"media-1:shot:{index:06d}",
        media_id="media-1",
        shot_index=index,
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
    )


def test_align_transcript_to_shots_by_overlap() -> None:
    aligned = align_transcript_to_shots(
        [
            TranscriptSegment(start_seconds=0.0, end_seconds=2.0, text="first"),
            TranscriptSegment(start_seconds=2.0, end_seconds=4.0, text="second"),
            TranscriptSegment(start_seconds=6.0, end_seconds=7.0, text="late"),
        ],
        [shot(0, 0.0, 3.0), shot(1, 3.0, 5.0)],
    )

    assert aligned[0].text == "first second"
    assert aligned[0].segment_count == 2
    assert aligned[1].text == "second"
    assert aligned[1].segment_count == 1


def test_resolve_whisper_compute_type() -> None:
    assert resolve_whisper_compute_type("auto", "cuda") == "float16"
    assert resolve_whisper_compute_type("auto", "cpu") == "int8"
    assert resolve_whisper_compute_type("float32", "cpu") == "float32"


async def test_faster_whisper_transcriber_lazy_loads_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Torch:
        cuda = Cuda()

    class FasterWhisper:
        class WhisperModel:
            def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
                calls.append(f"load:{model_name}:{device}:{compute_type}")

            def transcribe(self, path: str) -> tuple[list[object], None]:
                calls.append(f"transcribe:{path}")
                return [SimpleNamespace(start=1.0, end=2.0, text=" hello ")], None

    monkeypatch.setattr(transcript_module, "_import_torch", lambda: Torch())
    monkeypatch.setattr(transcript_module, "_import_faster_whisper", lambda: FasterWhisper)

    transcriber = FasterWhisperTranscriber(
        model_name="small",
        device="auto",
        compute_type="auto",
    )

    assert await transcriber.transcribe(tmp_path / "audio.wav") == [
        TranscriptSegment(start_seconds=1.0, end_seconds=2.0, text="hello")
    ]
    assert calls == [
        "load:small:cpu:int8",
        f"transcribe:{tmp_path / 'audio.wav'}",
    ]
