"""Transcript extraction and shot alignment for ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.visual_encoder import resolve_device


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class ShotTranscript:
    shot: DetectedShot
    text: str
    segment_count: int


class Transcriber(Protocol):
    async def transcribe(self, audio_path: Path) -> list[TranscriptSegment]: ...


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
    ) -> None:
        self._model_name = model_name
        self._configured_device = device
        self._configured_compute_type = compute_type
        self._model: Any | None = None

    async def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    def _transcribe_sync(self, audio_path: Path) -> list[TranscriptSegment]:
        self._load_model()
        assert self._model is not None
        segments, _ = self._model.transcribe(str(audio_path))
        return [
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=str(segment.text).strip(),
            )
            for segment in segments
            if str(segment.text).strip()
        ]

    def _load_model(self) -> None:
        if self._model is not None:
            return

        faster_whisper = _import_faster_whisper()
        torch = _import_torch()
        device = resolve_device(self._configured_device, torch)
        compute_type = resolve_whisper_compute_type(self._configured_compute_type, device)
        self._model = faster_whisper.WhisperModel(
            self._model_name,
            device=device,
            compute_type=compute_type,
        )


def align_transcript_to_shots(
    segments: list[TranscriptSegment],
    shots: list[DetectedShot],
) -> list[ShotTranscript]:
    aligned: list[ShotTranscript] = []
    for shot in shots:
        overlapping = [
            segment
            for segment in segments
            if overlaps(
                segment.start_seconds,
                segment.end_seconds,
                shot.start_seconds,
                shot.end_seconds,
            )
        ]
        text = " ".join(segment.text for segment in overlapping).strip()
        aligned.append(
            ShotTranscript(
                shot=shot,
                text=text,
                segment_count=len(overlapping),
            )
        )
    return aligned


def overlaps(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def resolve_whisper_compute_type(configured_compute_type: str, device: str) -> str:
    if configured_compute_type != "auto":
        return configured_compute_type
    return "float16" if device == "cuda" else "int8"


def _import_faster_whisper() -> Any:
    import faster_whisper

    return faster_whisper


def _import_torch() -> Any:
    import torch

    return torch
