from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kuvox_ai.modules.ingestion.visual_encoder as visual_encoder_module
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.visual_encoder import ClipVisualEncoder, resolve_device


def frame(path: Path) -> SampledFrame:
    return SampledFrame(
        shot=DetectedShot(
            shot_id="media-1:shot:000000",
            media_id="media-1",
            shot_index=0,
            start_seconds=0,
            end_seconds=2,
            duration_seconds=2,
        ),
        timestamp_seconds=1,
        path=path,
    )


def test_resolve_device_auto_prefers_cuda_when_available() -> None:
    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class Torch:
        cuda = Cuda()

    assert resolve_device("auto", Torch()) == "cuda"
    assert resolve_device("cpu", Torch()) == "cpu"


async def test_encoder_returns_empty_without_loading_model() -> None:
    encoder = ClipVisualEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="auto",
        batch_size=16,
    )

    assert await encoder.encode_frames([]) == []


async def test_encoder_lazy_loads_open_clip_and_batches_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Tensor:
        def __init__(self, values: list[list[float]] | None = None) -> None:
            self._values = values or [[1.0, 0.0]]

        def to(self, device: str) -> Tensor:
            calls.append(f"to:{device}")
            return self

        def norm(self, **_: object) -> Tensor:
            return self

        def clamp(self, **_: object) -> Tensor:
            return self

        def __truediv__(self, _: object) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def tolist(self) -> list[list[float]]:
            return self._values

    class Torch:
        cuda = Cuda()

        @staticmethod
        def no_grad() -> NoGrad:
            return NoGrad()

        @staticmethod
        def stack(images: list[Any]) -> Tensor:
            calls.append(f"stack:{len(images)}")
            return Tensor()

    class Image:
        @staticmethod
        def open(path: Path) -> Image:
            calls.append(f"open:{path.name}")
            return Image()

        def convert(self, mode: str) -> Image:
            calls.append(f"convert:{mode}")
            return self

    class Model:
        def to(self, device: str) -> Model:
            calls.append(f"model-to:{device}")
            return self

        def eval(self) -> None:
            calls.append("eval")

        def encode_image(self, _: Tensor) -> Tensor:
            calls.append("encode")
            return Tensor([[0.5, 0.5]])

    class OpenClip:
        @staticmethod
        def create_model_and_transforms(
            model_name: str,
            *,
            pretrained: str,
        ) -> tuple[Model, None, object]:
            calls.append(f"load:{model_name}:{pretrained}")

            def preprocess(_: Image) -> Tensor:
                calls.append("preprocess")
                return Tensor()

            return Model(), None, preprocess

    monkeypatch.setattr(visual_encoder_module, "_import_torch", lambda: Torch())
    monkeypatch.setattr(visual_encoder_module, "_import_pil_image", lambda: Image)
    monkeypatch.setattr(visual_encoder_module, "_import_open_clip", lambda: OpenClip())

    encoder = ClipVisualEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="auto",
        batch_size=1,
    )

    assert await encoder.encode_frames([frame(tmp_path / "one.jpg")]) == [[0.5, 0.5]]
    assert calls == [
        "load:ViT-B-32:laion2b_s34b_b79k",
        "model-to:cpu",
        "eval",
        "open:one.jpg",
        "convert:RGB",
        "preprocess",
        "stack:1",
        "to:cpu",
        "encode",
    ]
