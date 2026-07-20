from __future__ import annotations

import asyncio
import struct
from typing import cast

import pytest
from prometheus_client import generate_latest

from kuvox_ai.cache import CacheRead, CacheStore, ReadOutcome, WriteOutcome
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.modules.retrieval.query_embedding_cache import (
    QUERY_EMBEDDING_NORMALIZATION_ID,
    CachedQueryTextEmbeddingEncoder,
    QueryEmbeddingBinaryCodec,
    QueryEmbeddingCacheOutcome,
    canonicalize_query_text,
)
from kuvox_ai.text_embedding_cache import TEXT_EMBEDDING_MAGIC, TextEmbeddingBinaryCodec


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, bytes, int]] = []
        self.read_outcome: ReadOutcome | None = None
        self.write_outcome = WriteOutcome.SUCCESS
        self.raise_on_get: Exception | None = None
        self.raise_on_set: Exception | None = None
        self.get_delay = 0.0

    async def get(self, key: str) -> CacheRead:
        self.gets.append(key)
        if self.get_delay:
            await asyncio.sleep(self.get_delay)
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if self.read_outcome is not None:
            return CacheRead(self.read_outcome)
        value = self.values.get(key)
        return (
            CacheRead(ReadOutcome.HIT, value) if value is not None else CacheRead(ReadOutcome.MISS)
        )

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.sets.append((key, value, ttl_seconds))
        if self.write_outcome is WriteOutcome.SUCCESS:
            self.values[key] = value
        return self.write_outcome

    async def delete(self, key: str) -> WriteOutcome:
        self.values.pop(key, None)
        return WriteOutcome.SUCCESS


class RecordingEncoder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.vectors = vectors
        self.error: Exception | None = None

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.error is not None:
            raise self.error
        if self.vectors is not None:
            return self.vectors
        return [[float(index), float(index + 1), float(index + 2)] for index, _ in enumerate(texts)]


def cached(
    encoder: RecordingEncoder,
    cache: MemoryCache,
    *,
    enabled: bool = True,
    model_id: str = "model/test",
    dimension: int = 3,
    ttl_seconds: int = 604_800,
    legacy_read_enabled: bool = False,
) -> CachedQueryTextEmbeddingEncoder:
    return CachedQueryTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, encoder),
        cache=cast(CacheStore, cache),
        enabled=enabled,
        model_id=model_id,
        dimension=dimension,
        ttl_seconds=ttl_seconds,
        legacy_read_enabled=legacy_read_enabled,
    )


def shared_codec(*, model_id: str = "model/test", dimension: int = 3) -> TextEmbeddingBinaryCodec:
    return TextEmbeddingBinaryCodec(
        magic=TEXT_EMBEDDING_MAGIC,
        model_id=model_id,
        dimension=dimension,
    )


def test_canonicalization_normalizes_nfc_and_line_endings_only() -> None:
    assert canonicalize_query_text("Cafe\u0301\r\nline\rend") == "Café\nline\nend"
    assert canonicalize_query_text("Query") != canonicalize_query_text("query")
    assert canonicalize_query_text("a b") != canonicalize_query_text("a  b")
    assert canonicalize_query_text(" a ") == " a "


def test_binary_codec_round_trip_and_validation_categories() -> None:
    codec = shared_codec()
    encoded = codec.encode([0.25, -1.5, 2.0])
    assert encoded is not None
    assert encoded[:4] == b"KTEV"
    assert encoded[4] == 1
    assert struct.unpack(">I", encoded[5:9]) == (3,)
    decoded = codec.decode(encoded)
    assert decoded.outcome is None
    assert decoded.vector == pytest.approx([0.25, -1.5, 2.0])

    assert codec.decode(b"bad").outcome is QueryEmbeddingCacheOutcome.CORRUPT_DATA
    assert codec.decode(b"NOPE" + encoded[4:]).outcome is QueryEmbeddingCacheOutcome.CORRUPT_DATA
    assert (
        codec.decode(encoded[:4] + b"\x02" + encoded[5:]).outcome
        is QueryEmbeddingCacheOutcome.SCHEMA_MISMATCH
    )
    assert codec.decode(encoded[:-1]).outcome is QueryEmbeddingCacheOutcome.CORRUPT_DATA
    assert (
        shared_codec(model_id="other").decode(encoded).outcome
        is QueryEmbeddingCacheOutcome.IDENTITY_MISMATCH
    )
    assert (
        shared_codec(dimension=4).decode(encoded).outcome
        is QueryEmbeddingCacheOutcome.DIMENSION_MISMATCH
    )

    wrong_normalization = bytearray(encoded)
    wrong_normalization[41] ^= 0xFF
    assert (
        codec.decode(bytes(wrong_normalization)).outcome
        is QueryEmbeddingCacheOutcome.IDENTITY_MISMATCH
    )
    nan_payload = encoded[:73] + struct.pack(">3f", float("nan"), 0.0, 1.0)
    assert codec.decode(nan_payload).outcome is QueryEmbeddingCacheOutcome.NON_FINITE_DATA


def test_legacy_query_codec_retains_kqev_wire_format() -> None:
    encoded = QueryEmbeddingBinaryCodec(model_id="model/test", dimension=3).encode(
        [0.25, -0.5, 0.75]
    )
    assert encoded is not None
    assert encoded.startswith(b"KQEV\x01")


@pytest.mark.asyncio
async def test_empty_input_returns_without_encoder_or_cache_calls() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    assert await cached(encoder, store).encode_texts([]) == []
    assert encoder.calls == []
    assert store.gets == []


@pytest.mark.asyncio
async def test_disabled_mode_canonicalizes_records_bypass_and_never_calls_cache() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    result = await cached(encoder, store, enabled=False).encode_texts(["Cafe\u0301\rquery"])
    assert result == [[0.0, 1.0, 2.0]]
    assert encoder.calls == [["Café\nquery"]]
    assert store.gets == []
    assert store.sets == []


@pytest.mark.asyncio
async def test_miss_write_then_sequential_hit_uses_ttl_and_hashed_key() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    decorator = cached(encoder, store, ttl_seconds=123)

    first = await decorator.encode_texts(["private query"])
    second = await decorator.encode_texts(["private query"])

    assert first == second == [[0.0, 1.0, 2.0]]
    assert encoder.calls == [["private query"]]
    assert len(store.gets) == 2
    assert len(store.sets) == 1
    key, payload, ttl = store.sets[0]
    assert ttl == 123
    assert payload.startswith(b"KTEV\x01")
    assert key == decorator.key_for_text("private query")
    assert "private query" not in key
    assert key.startswith("kuvox:v1:ai:text-embedding:model:model/test:dim:3:norm:nfc-lf-v1:")


@pytest.mark.asyncio
async def test_legacy_query_hit_is_promoted_without_encoder_input() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    decorator = cached(encoder, store, ttl_seconds=321, legacy_read_enabled=True)
    legacy_payload = QueryEmbeddingBinaryCodec(model_id="model/test", dimension=3).encode(
        [0.25, -0.5, 0.75]
    )
    assert legacy_payload is not None
    store.values[decorator.legacy_key_for_text("legacy query")] = legacy_payload

    assert await decorator.encode_texts(["legacy query"]) == [[0.25, -0.5, 0.75]]
    assert encoder.calls == []
    assert store.gets == [
        decorator.key_for_text("legacy query"),
        decorator.legacy_key_for_text("legacy query"),
    ]
    assert store.sets[0][0] == decorator.key_for_text("legacy query")
    assert store.sets[0][1].startswith(b"KTEV\x01")
    assert store.sets[0][2] == 321


@pytest.mark.asyncio
async def test_disabled_legacy_read_recomputes_instead_of_reading_legacy_key() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    decorator = cached(encoder, store, legacy_read_enabled=False)
    legacy_payload = QueryEmbeddingBinaryCodec(model_id="model/test", dimension=3).encode(
        [0.25, -0.5, 0.75]
    )
    assert legacy_payload is not None
    store.values[decorator.legacy_key_for_text("legacy query")] = legacy_payload

    assert await decorator.encode_texts(["legacy query"]) == [[0.0, 1.0, 2.0]]
    assert encoder.calls == [["legacy query"]]
    assert store.gets == [decorator.key_for_text("legacy query")]


@pytest.mark.asyncio
async def test_equivalent_canonical_text_hits_but_case_and_whitespace_are_isolated() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    decorator = cached(encoder, store)

    assert decorator.key_for_text("Cafe\u0301\r\nline") == decorator.key_for_text("Café\nline")
    assert decorator.key_for_text("Query") != decorator.key_for_text("query")
    assert decorator.key_for_text("a b") != decorator.key_for_text("a  b")
    await decorator.encode_texts(["Cafe\u0301\r\nline"])
    await decorator.encode_texts(["Café\nline"])
    assert encoder.calls == [["Café\nline"]]


@pytest.mark.asyncio
async def test_mixed_batch_preserves_order_and_encodes_unresolved_once() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder(vectors=[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    decorator = cached(encoder, store)
    hit_payload = shared_codec().encode([0.0, 1.0, 2.0])
    assert hit_payload is not None
    store.values[decorator.key_for_text("hit")] = hit_payload

    result = await decorator.encode_texts(["hit", "miss-one", "miss-two"])

    assert result == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    assert encoder.calls == [["miss-one", "miss-two"]]
    assert [item[0] for item in store.sets] == [
        decorator.key_for_text("miss-one"),
        decorator.key_for_text("miss-two"),
    ]


@pytest.mark.asyncio
async def test_blank_strings_are_encoded_but_never_read_or_written() -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    result = await cached(encoder, store).encode_texts([" \r\n\t"])
    assert result == [[0.0, 1.0, 2.0]]
    assert encoder.calls == [[" \n\t"]]
    assert store.gets == []
    assert store.sets == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_value",
    [
        b"garbage",
        b"KTEV\x02" + b"\x00" * 100,
        b"KTEV\x01" + b"\x00" * 10,
    ],
)
async def test_invalid_cached_values_are_recomputed_and_repaired(bad_value: bytes) -> None:
    store = MemoryCache()
    encoder = RecordingEncoder()
    decorator = cached(encoder, store)
    key = decorator.key_for_text("repair")
    store.values[key] = bad_value

    assert await decorator.encode_texts(["repair"]) == [[0.0, 1.0, 2.0]]
    assert encoder.calls == [["repair"]]
    assert store.values[key].startswith(b"KTEV\x01")


@pytest.mark.asyncio
async def test_identity_dimension_and_non_finite_cached_values_are_repaired() -> None:
    scenarios = [
        shared_codec(model_id="other").encode([1.0, 2.0, 3.0]),
        shared_codec(dimension=4).encode([1.0, 2.0, 3.0, 4.0]),
    ]
    valid = shared_codec().encode([1.0, 2.0, 3.0])
    assert valid is not None
    scenarios.append(valid[:73] + struct.pack(">3f", float("inf"), 2.0, 3.0))

    for bad_value in scenarios:
        assert bad_value is not None
        store = MemoryCache()
        encoder = RecordingEncoder()
        decorator = cached(encoder, store)
        store.values[decorator.key_for_text("repair")] = bad_value
        assert await decorator.encode_texts(["repair"]) == [[0.0, 1.0, 2.0]]
        assert encoder.calls == [["repair"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("read_outcome", [ReadOutcome.ERROR, ReadOutcome.BYPASS])
async def test_cache_read_failure_or_bypass_falls_back(read_outcome: ReadOutcome) -> None:
    store = MemoryCache()
    store.read_outcome = read_outcome
    encoder = RecordingEncoder()
    assert await cached(encoder, store).encode_texts(["fallback"]) == [[0.0, 1.0, 2.0]]
    assert encoder.calls == [["fallback"]]


@pytest.mark.asyncio
async def test_store_exceptions_and_write_failures_are_fail_open() -> None:
    encoder = RecordingEncoder()
    read_failure = MemoryCache()
    read_failure.raise_on_get = TimeoutError("read timeout")
    assert await cached(encoder, read_failure).encode_texts(["read"]) == [[0.0, 1.0, 2.0]]

    write_failure = MemoryCache()
    write_failure.raise_on_set = ConnectionError("write failed")
    assert await cached(encoder, write_failure).encode_texts(["write"]) == [[0.0, 1.0, 2.0]]


@pytest.mark.asyncio
async def test_cancellation_and_encoder_exceptions_propagate() -> None:
    store = MemoryCache()
    store.get_delay = 60
    task = asyncio.create_task(cached(RecordingEncoder(), store).encode_texts(["cancel"]))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    encoder = RecordingEncoder()
    encoder.error = RuntimeError("authoritative encoder failed")
    with pytest.raises(RuntimeError, match="authoritative encoder failed"):
        await cached(encoder, MemoryCache()).encode_texts(["fail"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector",
    [
        [1.0, 2.0],
        [1.0, float("nan"), 3.0],
        [1.0, float("inf"), 3.0],
        [1.0, 3.5e38, 3.0],
    ],
)
async def test_invalid_authoritative_vectors_are_returned_but_not_cached(
    vector: list[float],
) -> None:
    store = MemoryCache()
    encoder = RecordingEncoder(vectors=[vector])
    assert await cached(encoder, store).encode_texts(["invalid"]) == [vector]
    assert store.sets == []


def test_key_isolates_model_dimension_and_normalization_identity() -> None:
    first = cached(RecordingEncoder(), MemoryCache(), model_id="model/a", dimension=3)
    second = cached(RecordingEncoder(), MemoryCache(), model_id="model/b", dimension=3)
    third = cached(RecordingEncoder(), MemoryCache(), model_id="model/a", dimension=4)
    assert first.key_for_text("query") != second.key_for_text("query")
    assert first.key_for_text("query") != third.key_for_text("query")
    assert f":norm:{QUERY_EMBEDDING_NORMALIZATION_ID}:" in first.key_for_text("query")
    assert first.key_for_text("query").replace("nfc-lf-v1", "old-norm") != first.key_for_text(
        "query"
    )


@pytest.mark.asyncio
async def test_prometheus_series_are_low_cardinality_and_do_not_expose_query_text() -> None:
    secret = "customer-secret-query"
    decorator = cached(RecordingEncoder(), MemoryCache())
    await decorator.encode_texts([secret])
    await decorator.encode_texts([secret])

    exposition = generate_latest().decode("utf-8")
    assert "kuvox_query_embedding_cache_operations_total" in exposition
    assert "kuvox_query_embedding_cache_duration_seconds" in exposition
    assert "kuvox_query_embedding_cache_payload_bytes" in exposition
    assert "kuvox_query_embedding_encoder_inputs_total" in exposition
    assert secret not in exposition
    assert "model=" not in "\n".join(
        line for line in exposition.splitlines() if "query_embedding" in line
    )
