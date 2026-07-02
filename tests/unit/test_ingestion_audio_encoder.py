from __future__ import annotations

from pathlib import Path

import pytest

import kuvox_ai.modules.ingestion.audio_encoder as audio_encoder_module
from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.audio_encoder import MsClapAudioEncoder, normalize_vectors
from kuvox_ai.modules.ingestion.models import DetectedShot


def clip(path: Path) -> ShotAudioClip:
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=2,
        duration_seconds=2,
    )
    return ShotAudioClip(
        shot=shot,
        path=path,
        clip_start_seconds=0,
        clip_end_seconds=2,
        clip_duration_seconds=2,
    )


def test_normalize_vectors() -> None:
    assert normalize_vectors([[3.0, 4.0]]) == [[0.6, 0.8]]


async def test_audio_encoder_returns_empty_without_loading_model() -> None:
    encoder = MsClapAudioEncoder(device="auto", embedding_dim=2, batch_size=16)

    assert await encoder.encode_audio_clips([]) == []


async def test_audio_encoder_lazy_loads_msclap(
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

    class Encoded:
        def tolist(self) -> list[list[float]]:
            return [[3.0, 4.0]]

    class MsClap:
        class CLAP:
            def __init__(self, *, use_cuda: bool) -> None:
                calls.append(f"load:{use_cuda}")

            def get_audio_embeddings(self, paths: list[str], *, resample: bool) -> Encoded:
                calls.append(f"encode:{Path(paths[0]).name}:{resample}")
                return Encoded()

    monkeypatch.setattr(audio_encoder_module, "_import_torch", lambda: Torch())
    monkeypatch.setattr(audio_encoder_module, "_import_msclap", lambda: MsClap)

    encoder = MsClapAudioEncoder(device="auto", embedding_dim=2, batch_size=1)

    assert await encoder.encode_audio_clips([clip(tmp_path / "shot.wav")]) == [[0.6, 0.8]]
    assert calls == ["load:False", "encode:shot.wav:True"]


async def test_audio_encoder_rejects_unexpected_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Torch:
        cuda = Cuda()

    class Encoded:
        def tolist(self) -> list[list[float]]:
            return [[1.0, 2.0]]

    class MsClap:
        class CLAP:
            def __init__(self, *, use_cuda: bool) -> None:
                return None

            def get_audio_embeddings(self, paths: list[str], *, resample: bool) -> Encoded:
                return Encoded()

    monkeypatch.setattr(audio_encoder_module, "_import_torch", lambda: Torch())
    monkeypatch.setattr(audio_encoder_module, "_import_msclap", lambda: MsClap)

    encoder = MsClapAudioEncoder(device="auto", embedding_dim=1024, batch_size=1)

    with pytest.raises(ValueError, match="dimension 2"):
        await encoder.encode_audio_clips([clip(tmp_path / "shot.wav")])
