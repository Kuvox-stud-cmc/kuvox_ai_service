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
        torch = _import_torch()
        embeddings: list[list[float]] = []
        for start in range(0, len(clips), self._batch_size):
            batch = clips[start : start + self._batch_size]
            preprocessed = _load_audio_batch(
                batch,
                model=self._model,
                torch=torch,
                device=self._device or "cpu",
            )
            encoded = self._model._get_audio_embeddings(preprocessed)
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


def _load_audio_batch(
    clips: list[ShotAudioClip],
    *,
    model: Any,
    torch: Any,
    device: str,
) -> Any:
    import librosa
    import numpy as np
    import soundfile as sf

    sample_rate = int(model.args.sampling_rate)
    sample_count = max(1, round(float(model.args.duration) * sample_rate))
    tensors: list[Any] = []

    for clip in clips:
        samples, source_rate = sf.read(
            clip.path,
            dtype="float32",
            always_2d=True,
        )
        mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
        if mono.size == 0:
            raise ValueError(f"Audio clip is empty: {clip.path}")
        if int(source_rate) != sample_rate:
            mono = librosa.resample(
                mono,
                orig_sr=int(source_rate),
                target_sr=sample_rate,
            )
        if mono.size < sample_count:
            mono = np.tile(mono, math.ceil(sample_count / mono.size))
        mono = np.asarray(mono[:sample_count], dtype=np.float32)
        tensors.append(torch.as_tensor(mono, dtype=torch.float32).reshape(1, -1))

    batch = torch.stack(tensors, dim=0)
    return batch.cuda() if device == "cuda" else batch


def _import_msclap() -> Any:
    import msclap

    return msclap


def _import_torch() -> Any:
    import torch

    return torch
