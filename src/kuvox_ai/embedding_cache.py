"""Shared binary codec for identity-bound embedding cache values."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import StrEnum

EMBEDDING_SCHEMA_VERSION = 1
_HEADER = struct.Struct(">4sBI32s32s")
_FLOAT32_BYTES = 4


class EmbeddingDecodeOutcome(StrEnum):
    CORRUPT_DATA = "corrupt_data"
    SCHEMA_MISMATCH = "schema_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    NON_FINITE_DATA = "non_finite_data"


@dataclass(frozen=True, slots=True)
class EmbeddingDecode:
    outcome: EmbeddingDecodeOutcome | None
    vector: list[float] | None = None


class EmbeddingBinaryCodec:
    """Versioned binary codec for one model- and pipeline-bound vector."""

    def __init__(
        self,
        *,
        magic: bytes,
        model_id: str,
        pipeline_id: str,
        dimension: int,
    ) -> None:
        if len(magic) != 4:
            raise ValueError("magic must be exactly four bytes")
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if not pipeline_id:
            raise ValueError("pipeline_id must be non-empty")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._magic = magic
        self._model_fingerprint = hashlib.sha256(model_id.encode("utf-8")).digest()
        self._pipeline_fingerprint = hashlib.sha256(pipeline_id.encode("utf-8")).digest()
        self._dimension = dimension

    def encode(self, vector: list[float]) -> bytes | None:
        if len(vector) != self._dimension:
            return None
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        try:
            payload = struct.pack(f">{self._dimension}f", *values)
        except (OverflowError, struct.error):
            return None
        if not all(
            math.isfinite(value) for value in struct.unpack(f">{self._dimension}f", payload)
        ):
            return None
        return (
            _HEADER.pack(
                self._magic,
                EMBEDDING_SCHEMA_VERSION,
                self._dimension,
                self._model_fingerprint,
                self._pipeline_fingerprint,
            )
            + payload
        )

    def decode(self, value: bytes) -> EmbeddingDecode:
        if len(value) < _HEADER.size:
            return EmbeddingDecode(EmbeddingDecodeOutcome.CORRUPT_DATA)
        try:
            magic, schema, dimension, model_fingerprint, pipeline_fingerprint = _HEADER.unpack_from(
                value
            )
        except struct.error:
            return EmbeddingDecode(EmbeddingDecodeOutcome.CORRUPT_DATA)
        if magic != self._magic:
            return EmbeddingDecode(EmbeddingDecodeOutcome.CORRUPT_DATA)
        if schema != EMBEDDING_SCHEMA_VERSION:
            return EmbeddingDecode(EmbeddingDecodeOutcome.SCHEMA_MISMATCH)
        expected_length = _HEADER.size + dimension * _FLOAT32_BYTES
        if len(value) != expected_length:
            return EmbeddingDecode(EmbeddingDecodeOutcome.CORRUPT_DATA)
        if (
            model_fingerprint != self._model_fingerprint
            or pipeline_fingerprint != self._pipeline_fingerprint
        ):
            return EmbeddingDecode(EmbeddingDecodeOutcome.IDENTITY_MISMATCH)
        if dimension != self._dimension:
            return EmbeddingDecode(EmbeddingDecodeOutcome.DIMENSION_MISMATCH)
        try:
            vector = list(struct.unpack_from(f">{dimension}f", value, _HEADER.size))
        except struct.error:
            return EmbeddingDecode(EmbeddingDecodeOutcome.CORRUPT_DATA)
        if not all(math.isfinite(item) for item in vector):
            return EmbeddingDecode(EmbeddingDecodeOutcome.NON_FINITE_DATA)
        return EmbeddingDecode(None, vector)
