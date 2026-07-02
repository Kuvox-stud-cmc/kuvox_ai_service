"""Audio embedding for shot-level audio clips."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Protocol, cast

from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.visual_encoder import resolve_device


class AudioEmbeddingEncoder(Protocol):
    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]: ...


class MsClapAudioEncoder:
    def __init__(
        self,
        *,
        device: str,
        embedding_dim: int,
        batch_size: int,
    ) -> None:
        self._configured_device = device
        self._embedding_dim = embedding_dim
        self._batch_size = batch_size
        self._model: Any | None = None
        self._device: str | None = None

    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        if not clips:
            return []
        return await asyncio.to_thread(self._encode_audio_clips_sync, clips)

    def _encode_audio_clips_sync(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        self._load_model()
        assert self._model is not None
        embeddings: list[list[float]] = []
        for start in range(0, len(clips), self._batch_size):
            batch = clips[start : start + self._batch_size]
            encoded = self._model.get_audio_embeddings(
                [str(clip.path) for clip in batch],
                resample=True,
            )
            vectors = cast(list[list[float]], encoded.tolist())
            embeddings.extend(normalize_vectors(vectors))

        for index, embedding in enumerate(embeddings):
            if len(embedding) != self._embedding_dim:
                raise ValueError(
                    f"Audio embedding {index} has dimension {len(embedding)}, "
                    f"expected {self._embedding_dim}."
                )
        return embeddings

    def _load_model(self) -> None:
        if self._model is not None:
            return

        msclap = _import_msclap()
        torch = _import_torch()
        self._device = resolve_device(self._configured_device, torch)
        self._model = msclap.CLAP(use_cuda=self._device == "cuda")


def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            normalized.append(vector)
        else:
            normalized.append([value / norm for value in vector])
    return normalized


def _import_msclap() -> Any:
    import msclap

    return msclap


def _import_torch() -> Any:
    import torch

    return torch
