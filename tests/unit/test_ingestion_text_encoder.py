from __future__ import annotations

import pytest

import kuvox_ai.modules.ingestion.text_encoder as text_encoder_module
from kuvox_ai.modules.ingestion.text_encoder import SentenceTransformerTextEncoder


async def test_text_encoder_returns_empty_without_loading_model() -> None:
    encoder = SentenceTransformerTextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="auto",
        batch_size=32,
    )

    assert await encoder.encode_texts([]) == []


async def test_text_encoder_lazy_loads_sentence_transformer(
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
            return [[0.1, 0.2]]

    class SentenceTransformers:
        class SentenceTransformer:
            def __init__(self, model_name: str, *, device: str) -> None:
                calls.append(f"load:{model_name}:{device}")

            def encode(
                self,
                texts: list[str],
                *,
                batch_size: int,
                normalize_embeddings: bool,
                convert_to_numpy: bool,
            ) -> Encoded:
                calls.append(
                    f"encode:{texts[0]}:{batch_size}:{normalize_embeddings}:{convert_to_numpy}"
                )
                return Encoded()

    monkeypatch.setattr(text_encoder_module, "_import_torch", lambda: Torch())
    monkeypatch.setattr(
        text_encoder_module,
        "_import_sentence_transformers",
        lambda: SentenceTransformers,
    )

    encoder = SentenceTransformerTextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="auto",
        batch_size=16,
    )

    assert await encoder.encode_texts(["hello"]) == [[0.1, 0.2]]
    assert calls == [
        "load:sentence-transformers/all-MiniLM-L6-v2:cpu",
        "encode:hello:16:True:True",
    ]
