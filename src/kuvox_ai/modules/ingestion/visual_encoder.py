"""CLIP visual embedding for sampled shot frames."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame


class VisualEncoder(Protocol):
    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]: ...


class ClipVisualEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: str,
        device: str,
        batch_size: int,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._configured_device = device
        self._batch_size = batch_size
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._device: str | None = None

    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
        if not frames:
            return []
        return await asyncio.to_thread(self._encode_frames_sync, frames)

    def _encode_frames_sync(self, frames: list[SampledFrame]) -> list[list[float]]:
        torch = _import_torch()
        pil_image = _import_pil_image()
        self._load_model()
        assert self._model is not None
        assert self._preprocess is not None
        assert self._device is not None

        embeddings: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(frames), self._batch_size):
                batch_frames = frames[start : start + self._batch_size]
                images = [
                    self._preprocess(pil_image.open(frame.path).convert("RGB"))
                    for frame in batch_frames
                ]
                tensor = torch.stack(images).to(self._device)
                encoded = self._model.encode_image(tensor)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                embeddings.extend(cast(list[list[float]], encoded.cpu().tolist()))
        return embeddings

    def _load_model(self) -> None:
        if self._model is not None:
            return

        open_clip = _import_open_clip()
        torch = _import_torch()
        self._device = resolve_device(self._configured_device, torch)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name,
            pretrained=self._pretrained,
        )
        self._model = model.to(self._device)
        self._model.eval()
        self._preprocess = preprocess


def resolve_device(configured_device: str, torch: Any) -> str:
    if configured_device != "auto":
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _import_open_clip() -> Any:
    import open_clip

    return open_clip


def _import_torch() -> Any:
    import torch

    return torch


def _import_pil_image() -> Any:
    from PIL import Image

    return Image
