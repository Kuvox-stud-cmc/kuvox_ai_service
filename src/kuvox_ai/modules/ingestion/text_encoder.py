"""Sentence embedding for transcript and OCR text."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from kuvox_ai.modules.ingestion.visual_encoder import resolve_device


class TextEmbeddingEncoder(Protocol):
    async def encode_texts(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerTextEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        self._model_name = model_name
        self._configured_device = device
        self._batch_size = batch_size
        self._model: Any | None = None

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_texts_sync, texts)

    def _encode_texts_sync(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        assert self._model is not None
        encoded = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return cast(list[list[float]], encoded.tolist())

    def _load_model(self) -> None:
        if self._model is not None:
            return

        sentence_transformers = _import_sentence_transformers()
        torch = _import_torch()
        device = resolve_device(self._configured_device, torch)
        self._model = sentence_transformers.SentenceTransformer(
            self._model_name,
            device=device,
        )


def _import_sentence_transformers() -> Any:
    import sentence_transformers

    return sentence_transformers


def _import_torch() -> Any:
    import torch

    return torch
