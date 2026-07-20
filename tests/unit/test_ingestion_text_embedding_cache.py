from __future__ import annotations

from typing import cast

import pytest
from prometheus_client import generate_latest

from kuvox_ai.cache import CacheRead, CacheStore, CacheWrite, ReadOutcome, WriteOutcome
from kuvox_ai.modules.ingestion.text_embedding_cache import (
    CachedIngestionTextEmbeddingEncoder,
)
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.modules.retrieval.query_embedding_cache import (
    CachedQueryTextEmbeddingEncoder,
    QueryEmbeddingBinaryCodec,
)
from kuvox_ai.text_embedding_cache import TEXT_EMBEDDING_MAGIC, TextEmbeddingBinaryCodec


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, bytes, int]] = []

    async def get(self, key: str) -> CacheRead:
        self.gets.append(key)
        value = self.values.get(key)
        return (
            CacheRead(ReadOutcome.HIT, value) if value is not None else CacheRead(ReadOutcome.MISS)
        )

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        self.sets.append((key, value, ttl_seconds))
        self.values[key] = value
        return WriteOutcome.SUCCESS

    async def delete(self, key: str) -> WriteOutcome:
        self.values.pop(key, None)
        return WriteOutcome.SUCCESS


class BulkMemoryCache(MemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_gets: list[list[str]] = []
        self.bulk_sets: list[list[CacheWrite]] = []

    async def get_many(self, keys: list[str]) -> list[CacheRead]:
        self.bulk_gets.append(list(keys))
        return [
            CacheRead(ReadOutcome.HIT, self.values[key])
            if key in self.values
            else CacheRead(ReadOutcome.MISS)
            for key in keys
        ]

    async def set_many(self, entries: list[CacheWrite]) -> list[WriteOutcome]:
        self.bulk_sets.append(list(entries))
        for entry in entries:
            self.values[entry.key] = entry.value
        return [WriteOutcome.SUCCESS for _ in entries]


class RecordingEncoder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vector for _ in texts]


def ingestion_cached(
    encoder: RecordingEncoder,
    cache: MemoryCache,
    *,
    enabled: bool = True,
) -> CachedIngestionTextEmbeddingEncoder:
    return CachedIngestionTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, encoder),
        cache=cast(CacheStore, cache),
        enabled=enabled,
        model_id="model/shared",
        dimension=3,
        ttl_seconds=604_800,
        legacy_read_enabled=False,
    )


def query_cached(
    encoder: RecordingEncoder,
    cache: MemoryCache,
) -> CachedQueryTextEmbeddingEncoder:
    return CachedQueryTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, encoder),
        cache=cast(CacheStore, cache),
        enabled=True,
        model_id="model/shared",
        dimension=3,
        ttl_seconds=604_800,
        legacy_read_enabled=False,
    )


@pytest.mark.asyncio
async def test_ingestion_warmup_is_reused_by_query_consumer() -> None:
    store = MemoryCache()
    ingestion_encoder = RecordingEncoder([0.1, 0.2, 0.3])
    query_encoder = RecordingEncoder([9.0, 9.0, 9.0])
    ingestion = ingestion_cached(ingestion_encoder, store)
    query = query_cached(query_encoder, store)

    first = await ingestion.encode_texts(["shared transcript"])
    second = await query.encode_texts(["shared transcript"])

    assert second[0] == pytest.approx(first[0])
    assert ingestion_encoder.calls == [["shared transcript"]]
    assert query_encoder.calls == []
    assert ingestion.key_for_text("shared transcript") == query.key_for_text("shared transcript")
    assert store.sets[0][1].startswith(b"KTEV\x01")
    assert store.sets[0][2] == 604_800


@pytest.mark.asyncio
async def test_query_warmup_is_reused_by_ingestion_consumer() -> None:
    store = MemoryCache()
    query_encoder = RecordingEncoder([0.4, 0.5, 0.6])
    ingestion_encoder = RecordingEncoder([9.0, 9.0, 9.0])
    query = query_cached(query_encoder, store)
    ingestion = ingestion_cached(ingestion_encoder, store)

    first = await query.encode_texts(["shared OCR"])
    second = await ingestion.encode_texts(["shared OCR"])

    assert second[0] == pytest.approx(first[0])
    assert query_encoder.calls == [["shared OCR"]]
    assert ingestion_encoder.calls == []


@pytest.mark.asyncio
async def test_ingestion_consumer_promotes_legacy_query_entry() -> None:
    store = MemoryCache()
    authoritative = RecordingEncoder([9.0, 9.0, 9.0])
    ingestion = CachedIngestionTextEmbeddingEncoder(
        authoritative,
        cache=cast(CacheStore, store),
        enabled=True,
        model_id="model/shared",
        dimension=3,
        ttl_seconds=604_800,
        legacy_read_enabled=True,
    )
    legacy_payload = QueryEmbeddingBinaryCodec(model_id="model/shared", dimension=3).encode(
        [0.2, 0.4, 0.6]
    )
    assert legacy_payload is not None
    store.values[ingestion.legacy_key_for_text("legacy transcript")] = legacy_payload

    result = await ingestion.encode_texts(["legacy transcript"])

    assert result[0] == pytest.approx([0.2, 0.4, 0.6])
    assert authoritative.calls == []
    assert store.values[ingestion.key_for_text("legacy transcript")].startswith(b"KTEV\x01")


@pytest.mark.asyncio
async def test_ingestion_disabled_bypasses_store_and_still_canonicalizes() -> None:
    store = MemoryCache()
    authoritative = RecordingEncoder([0.1, 0.2, 0.3])
    encoder = ingestion_cached(authoritative, store, enabled=False)

    assert await encoder.encode_texts(["Cafe\u0301\rtext"]) == [[0.1, 0.2, 0.3]]
    assert authoritative.calls == [["Café\ntext"]]
    assert store.gets == []
    assert store.sets == []


@pytest.mark.asyncio
async def test_bulk_partial_hits_duplicates_legacy_promotion_and_order() -> None:
    store = BulkMemoryCache()
    authoritative = RecordingEncoder([0.1, 0.2, 0.3])
    encoder = CachedIngestionTextEmbeddingEncoder(
        authoritative,
        cache=cast(CacheStore, store),
        enabled=True,
        model_id="model/shared",
        dimension=3,
        ttl_seconds=604_800,
        legacy_read_enabled=True,
    )
    shared_codec = TextEmbeddingBinaryCodec(
        magic=TEXT_EMBEDDING_MAGIC,
        model_id="model/shared",
        dimension=3,
    )
    hit_payload = shared_codec.encode([1.0, 2.0, 3.0])
    legacy_payload = QueryEmbeddingBinaryCodec(model_id="model/shared", dimension=3).encode(
        [4.0, 5.0, 6.0]
    )
    assert hit_payload is not None
    assert legacy_payload is not None
    store.values[encoder.key_for_text("hit")] = hit_payload
    store.values[encoder.legacy_key_for_text("legacy")] = legacy_payload

    result = await encoder.encode_texts(["miss", "hit", "miss", "legacy", " \r\n\t"])

    assert result == [
        [0.1, 0.2, 0.3],
        pytest.approx([1.0, 2.0, 3.0]),
        [0.1, 0.2, 0.3],
        pytest.approx([4.0, 5.0, 6.0]),
        [0.1, 0.2, 0.3],
    ]
    assert authoritative.calls == [["miss", " \n\t"]]
    assert store.bulk_gets == [
        [
            encoder.key_for_text("miss"),
            encoder.key_for_text("hit"),
            encoder.key_for_text("legacy"),
        ],
        [
            encoder.legacy_key_for_text("miss"),
            encoder.legacy_key_for_text("legacy"),
        ],
    ]
    assert len(store.bulk_sets) == 1
    assert [entry.key for entry in store.bulk_sets[0]] == [
        encoder.key_for_text("legacy"),
        encoder.key_for_text("miss"),
    ]
    assert store.gets == []
    assert store.sets == []


@pytest.mark.asyncio
async def test_complete_authoritative_text_vector_count_is_validated_before_writes() -> None:
    class TooManyEncoder:
        async def encode_texts(self, texts: list[str]) -> list[list[float]]:
            del texts
            return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    store = BulkMemoryCache()
    encoder = CachedIngestionTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, TooManyEncoder()),
        cache=cast(CacheStore, store),
        enabled=True,
        model_id="model/shared",
        dimension=3,
        legacy_read_enabled=False,
    )

    with pytest.raises(ValueError, match="unexpected number"):
        await encoder.encode_texts(["one"])
    assert store.bulk_sets == []


@pytest.mark.asyncio
async def test_ingestion_metrics_have_no_sensitive_or_high_cardinality_labels() -> None:
    secret = "private transcript belonging to a customer"
    encoder = ingestion_cached(RecordingEncoder([0.1, 0.2, 0.3]), MemoryCache())
    await encoder.encode_texts([secret])
    await encoder.encode_texts([secret])

    exposition = generate_latest().decode("utf-8")
    assert "kuvox_ingestion_text_embedding_cache_operations_total" in exposition
    assert "kuvox_ingestion_text_embedding_cache_duration_seconds" in exposition
    assert "kuvox_ingestion_text_embedding_cache_payload_bytes" in exposition
    assert "kuvox_ingestion_text_embedding_encoder_inputs_total" in exposition
    assert secret not in exposition
    ingestion_lines = "\n".join(
        line for line in exposition.splitlines() if "ingestion_text_embedding" in line
    )
    for forbidden in ("model=", "media=", "owner=", "user=", "hash=", "query="):
        assert forbidden not in ingestion_lines
