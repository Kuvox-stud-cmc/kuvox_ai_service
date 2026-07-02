from __future__ import annotations

from pathlib import Path

import pytest

import kuvox_ai.modules.ingestion.ocr as ocr_module
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.ocr import EasyOcrReader, parse_easyocr_item, resolve_ocr_gpu


def frame(path: Path) -> SampledFrame:
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=2,
        duration_seconds=2,
    )
    return SampledFrame(shot=shot, timestamp_seconds=1.0, path=path)


def test_parse_easyocr_item() -> None:
    assert parse_easyocr_item([[], "hello", 0.75]) == ("hello", 0.75)
    assert parse_easyocr_item(["bad"]) is None


def test_resolve_ocr_gpu_explicit_values() -> None:
    assert resolve_ocr_gpu("true") is True
    assert resolve_ocr_gpu("false") is False


async def test_easyocr_reader_filters_by_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class EasyOcr:
        class Reader:
            def __init__(self, languages: list[str], *, gpu: bool) -> None:
                calls.append(f"load:{','.join(languages)}:{gpu}")

            def readtext(self, path: str) -> list[object]:
                calls.append(f"read:{Path(path).name}")
                return [
                    ([], "LOW", 0.2),
                    ([], "Hello", 0.8),
                    ([], "World", 0.6),
                ]

    monkeypatch.setattr(ocr_module, "_import_easyocr", lambda: EasyOcr)

    reader = EasyOcrReader(languages=["en"], gpu="false", min_confidence=0.3)
    result = await reader.read_frames([frame(tmp_path / "shot.jpg")])

    assert result[0].text == "Hello World"
    assert result[0].text_block_count == 2
    assert result[0].mean_confidence == pytest.approx(0.7)
    assert calls == ["load:en:False", "read:shot.jpg"]
