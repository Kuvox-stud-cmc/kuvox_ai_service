"""OCR extraction from sampled shot frames."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot


@dataclass(frozen=True, slots=True)
class ShotOcrText:
    shot: DetectedShot
    text: str
    frame_timestamp_seconds: float
    text_block_count: int
    mean_confidence: float | None


class OcrReader(Protocol):
    async def read_frames(self, frames: list[SampledFrame]) -> list[ShotOcrText]: ...


class EasyOcrReader:
    def __init__(
        self,
        *,
        languages: list[str],
        gpu: str,
        min_confidence: float,
    ) -> None:
        self._languages = languages
        self._configured_gpu = gpu
        self._min_confidence = min_confidence
        self._reader: Any | None = None

    async def read_frames(self, frames: list[SampledFrame]) -> list[ShotOcrText]:
        if not frames:
            return []
        return await asyncio.to_thread(self._read_frames_sync, frames)

    def _read_frames_sync(self, frames: list[SampledFrame]) -> list[ShotOcrText]:
        self._load_reader()
        assert self._reader is not None

        results: list[ShotOcrText] = []
        for frame in frames:
            text_blocks: list[str] = []
            confidences: list[float] = []
            for item in self._reader.readtext(str(frame.path)):
                parsed = parse_easyocr_item(item)
                if parsed is None:
                    continue
                text, confidence = parsed
                if confidence < self._min_confidence or not text.strip():
                    continue
                text_blocks.append(text.strip())
                confidences.append(confidence)

            mean_confidence = sum(confidences) / len(confidences) if confidences else None
            results.append(
                ShotOcrText(
                    shot=frame.shot,
                    text=" ".join(text_blocks).strip(),
                    frame_timestamp_seconds=frame.timestamp_seconds,
                    text_block_count=len(text_blocks),
                    mean_confidence=mean_confidence,
                )
            )
        return results

    def _load_reader(self) -> None:
        if self._reader is not None:
            return

        easyocr = _import_easyocr()
        self._reader = easyocr.Reader(
            self._languages,
            gpu=resolve_ocr_gpu(self._configured_gpu),
        )


def parse_easyocr_item(item: Any) -> tuple[str, float] | None:
    if not isinstance(item, (tuple, list)) or len(item) < 3:
        return None
    text = item[1]
    confidence = item[2]
    if not isinstance(text, str):
        return None
    try:
        return text, float(confidence)
    except (TypeError, ValueError):
        return None


def resolve_ocr_gpu(configured_gpu: str) -> bool:
    if configured_gpu.lower() in {"true", "1", "yes"}:
        return True
    if configured_gpu.lower() in {"false", "0", "no"}:
        return False
    torch = _import_torch()
    return bool(torch.cuda.is_available())


def _import_easyocr() -> Any:
    import easyocr

    return easyocr


def _import_torch() -> Any:
    import torch

    return torch
