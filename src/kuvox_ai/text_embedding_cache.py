"""Shared cache implementation for deterministic text embeddings."""

from __future__ import annotations

import asyncio
import random
import time
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from prometheus_client import Counter, Histogram

from kuvox_ai.cache import (
    CacheKeyFactory,
    CacheRead,
    CacheStore,
    CacheWrite,
    ReadOutcome,
    WriteOutcome,
    cache_get_many,
    cache_set_many,
)
from kuvox_ai.distributed_lock import LockAcquireOutcome, LockHandle, LockStore
from kuvox_ai.embedding_cache import EmbeddingBinaryCodec, EmbeddingDecodeOutcome
from kuvox_ai.metrics import SINGLE_FLIGHT_EVENTS, SINGLE_FLIGHT_HELD_LOCKS, SINGLE_FLIGHT_WAIT
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder

TEXT_EMBEDDING_NORMALIZATION_ID = "nfc-lf-v1"
TEXT_EMBEDDING_MAGIC = b"KTEV"
LEGACY_QUERY_EMBEDDING_MAGIC = b"KQEV"


class TextEmbeddingCacheOutcome(StrEnum):
    HIT = "hit"
    LEGACY_HIT = "legacy_hit"
    MISS = "miss"
    BYPASS = "bypass"
    REDIS_ERROR = "redis_error"
    CORRUPT_DATA = "corrupt_data"
    SCHEMA_MISMATCH = "schema_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    NON_FINITE_DATA = "non_finite_data"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class TextEmbeddingDecode:
    outcome: TextEmbeddingCacheOutcome | None
    vector: list[float] | None = None


@dataclass(frozen=True, slots=True)
class TextEmbeddingCacheMetrics:
    operations: Counter
    duration: Histogram
    payload_bytes: Histogram
    encoder_inputs: Counter


@dataclass(slots=True)
class _TextGroup:
    text: str
    positions: list[int]
    cacheable: bool
    miss_outcome: TextEmbeddingCacheOutcome


def canonicalize_text_embedding_input(text: str) -> str:
    """Normalize only Unicode composition and line-ending representation."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


class TextEmbeddingBinaryCodec:
    """Versioned, identity-bound binary codec for one text embedding."""

    def __init__(self, *, magic: bytes, model_id: str, dimension: int) -> None:
        self._codec = EmbeddingBinaryCodec(
            magic=magic,
            model_id=model_id,
            pipeline_id=TEXT_EMBEDDING_NORMALIZATION_ID,
            dimension=dimension,
        )

    def encode(self, vector: list[float]) -> bytes | None:
        return self._codec.encode(vector)

    def decode(self, value: bytes) -> TextEmbeddingDecode:
        decoded = self._codec.decode(value)
        outcome = (
            TextEmbeddingCacheOutcome(decoded.outcome.value)
            if isinstance(decoded.outcome, EmbeddingDecodeOutcome)
            else None
        )
        return TextEmbeddingDecode(outcome, decoded.vector)


class CachedTextEmbeddingEncoder:
    """Shared cache-aware decorator with optional legacy query-key promotion."""

    def __init__(
        self,
        wrapped: TextEmbeddingEncoder,
        *,
        cache: CacheStore,
        metrics: TextEmbeddingCacheMetrics,
        enabled: bool,
        model_id: str,
        dimension: int,
        ttl_seconds: int = 604_800,
        key_prefix: str = "kuvox:v1",
        legacy_read_enabled: bool = True,
        lock_store: LockStore | None = None,
        single_flight_enabled: bool = False,
        lock_ttl_seconds: float = 30,
        lock_wait_seconds: float = 15,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._wrapped = wrapped
        self._cache = cache
        self._metrics = metrics
        self._enabled = enabled
        self._dimension = dimension
        self._ttl_seconds = ttl_seconds
        self._keys = CacheKeyFactory(key_prefix)
        self._model_id = model_id
        self._legacy_read_enabled = legacy_read_enabled
        self._lock_store = lock_store
        self._single_flight_enabled = single_flight_enabled and lock_store is not None
        self._lock_ttl_seconds = lock_ttl_seconds
        self._lock_wait_seconds = lock_wait_seconds
        self._lock_poll_seconds = lock_poll_seconds
        self._codec = TextEmbeddingBinaryCodec(
            magic=TEXT_EMBEDDING_MAGIC,
            model_id=model_id,
            dimension=dimension,
        )
        self._legacy_codec = TextEmbeddingBinaryCodec(
            magic=LEGACY_QUERY_EMBEDDING_MAGIC,
            model_id=model_id,
            dimension=dimension,
        )

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        canonical_texts = [canonicalize_text_embedding_input(text) for text in texts]
        if not self._enabled:
            self._record_inputs(TextEmbeddingCacheOutcome.BYPASS, len(canonical_texts))
            self._record_operations(TextEmbeddingCacheOutcome.BYPASS, len(canonical_texts))
            return await self._wrapped.encode_texts(canonical_texts)

        resolved: list[list[float] | None] = [None] * len(canonical_texts)
        cacheable_by_text: dict[str, _TextGroup] = {}
        authoritative_groups: list[_TextGroup] = []
        for position, text in enumerate(canonical_texts):
            if not text.strip():
                authoritative_groups.append(
                    _TextGroup(
                        text=text,
                        positions=[position],
                        cacheable=False,
                        miss_outcome=TextEmbeddingCacheOutcome.BYPASS,
                    )
                )
                continue
            group = cacheable_by_text.get(text)
            if group is None:
                group = _TextGroup(
                    text=text,
                    positions=[],
                    cacheable=True,
                    miss_outcome=TextEmbeddingCacheOutcome.MISS,
                )
                cacheable_by_text[text] = group
            group.positions.append(position)

        cacheable_groups = list(cacheable_by_text.values())
        reads = await self._read_many(
            [self.key_for_canonical_text(group.text) for group in cacheable_groups],
            self._codec,
            operation="read",
        )
        legacy_candidates: list[_TextGroup] = []
        for group, (outcome, vector) in zip(cacheable_groups, reads, strict=True):
            if vector is not None:
                self._record_operations(TextEmbeddingCacheOutcome.HIT, len(group.positions))
                self._resolve_group(resolved, group, vector)
                continue
            group.miss_outcome = outcome
            if outcome is TextEmbeddingCacheOutcome.MISS and self._legacy_read_enabled:
                legacy_candidates.append(group)

        writes: list[CacheWrite] = []
        if legacy_candidates:
            legacy_reads = await self._read_many(
                [self.legacy_key_for_canonical_text(group.text) for group in legacy_candidates],
                self._legacy_codec,
                operation="legacy_read",
            )
            for group, (outcome, vector) in zip(legacy_candidates, legacy_reads, strict=True):
                if vector is not None:
                    self._record_operations(
                        TextEmbeddingCacheOutcome.LEGACY_HIT,
                        len(group.positions),
                    )
                    self._resolve_group(resolved, group, vector)
                    write = self._prepare_write(group.text, vector)
                    if write is not None:
                        writes.append(write)
                    continue
                group.miss_outcome = outcome

        for group in cacheable_groups:
            if resolved[group.positions[0]] is not None:
                continue
            self._record_operations(group.miss_outcome, len(group.positions))
            authoritative_groups.append(group)

        authoritative_groups.sort(key=lambda group: group.positions[0])
        if authoritative_groups:
            if self._single_flight_enabled:
                await self._write_many(writes)
                await self._encode_single_flight(authoritative_groups, resolved)
            else:
                await self._compute_groups(authoritative_groups, resolved, writes)
        else:
            await self._write_many(writes)
        if any(vector is None for vector in resolved):
            raise ValueError("Text embedding encoder returned an unexpected number of vectors")
        return [vector for vector in resolved if vector is not None]

    async def _encode_single_flight(
        self,
        groups: list[_TextGroup],
        resolved: list[list[float] | None],
    ) -> None:
        assert self._lock_store is not None
        direct = [group for group in groups if not group.cacheable]
        local: list[_TextGroup] = []
        owned: list[tuple[_TextGroup, LockHandle]] = []
        contended: dict[str, tuple[_TextGroup, LockHandle]] = {}

        for group in (group for group in groups if group.cacheable):
            if group.miss_outcome in {
                TextEmbeddingCacheOutcome.REDIS_ERROR,
                TextEmbeddingCacheOutcome.BYPASS,
            }:
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "bypass").inc()
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "authoritative_fallback").inc()
                local.append(group)
                continue
            cache_key = self.key_for_canonical_text(group.text)
            try:
                attempt = await self._lock_store.acquire(
                    "query-embedding", cache_key, self._lock_ttl_seconds
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - optional coordination fails open
                attempt = None
            if attempt is None:
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "acquisition_error").inc()
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "authoritative_fallback").inc()
                local.append(group)
                continue
            if attempt.outcome is LockAcquireOutcome.ACQUIRED and attempt.handle is not None:
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "leader").inc()
                owned.append((group, attempt.handle))
            elif attempt.outcome is LockAcquireOutcome.CONTENDED and attempt.handle is not None:
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "join").inc()
                contended[group.text] = (group, attempt.handle)
            else:
                outcome = (
                    "bypass"
                    if attempt.outcome is LockAcquireOutcome.BYPASS
                    else "acquisition_error"
                )
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", outcome).inc()
                SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "authoritative_fallback").inc()
                local.append(group)

        await self._compute_owned_and_local([*direct, *local], owned, resolved)
        if not contended:
            return

        wait_started = time.perf_counter()
        deadline = time.monotonic() + self._lock_wait_seconds
        while contended and time.monotonic() < deadline:
            await asyncio.sleep(max(0.001, self._lock_poll_seconds * random.uniform(0.8, 1.2)))
            pending = list(contended.values())
            reads = await self._read_many(
                [self.key_for_canonical_text(group.text) for group, _handle in pending],
                self._codec,
                operation="single_flight_poll",
            )
            retry_owned: list[tuple[_TextGroup, LockHandle]] = []
            retry_local: list[_TextGroup] = []
            for (group, handle), (outcome, vector) in zip(pending, reads, strict=True):
                if vector is not None:
                    self._resolve_group(resolved, group, vector)
                    contended.pop(group.text, None)
                    SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "joined_cache_hit").inc()
                    continue
                if outcome in {
                    TextEmbeddingCacheOutcome.REDIS_ERROR,
                    TextEmbeddingCacheOutcome.BYPASS,
                }:
                    contended.pop(group.text, None)
                    retry_local.append(group)
                    SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "bypass").inc()
                    continue
                try:
                    locked = await self._lock_store.is_locked(handle.key)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - optional coordination fails open
                    locked = None
                if locked is None:
                    contended.pop(group.text, None)
                    retry_local.append(group)
                    SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "bypass").inc()
                    continue
                if locked:
                    continue
                try:
                    attempt = await self._lock_store.acquire(
                        "query-embedding",
                        self.key_for_canonical_text(group.text),
                        self._lock_ttl_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - optional coordination fails open
                    attempt = None
                if attempt is None:
                    contended.pop(group.text, None)
                    retry_local.append(group)
                    SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "acquisition_error").inc()
                    continue
                if attempt.outcome is LockAcquireOutcome.ACQUIRED and attempt.handle is not None:
                    contended.pop(group.text, None)
                    retry_owned.append((group, attempt.handle))
                    SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "leader").inc()
                elif attempt.outcome in {LockAcquireOutcome.BYPASS, LockAcquireOutcome.ERROR}:
                    contended.pop(group.text, None)
                    retry_local.append(group)
            if retry_owned or retry_local:
                await self._compute_owned_and_local(retry_local, retry_owned, resolved)

        if contended:
            timed_out = [group for group, _handle in contended.values()]
            SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "timeout").inc(len(timed_out))
            SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "authoritative_fallback").inc(
                len(timed_out)
            )
            await self._compute_groups(timed_out, resolved)
        SINGLE_FLIGHT_WAIT.labels("ai", "query-embedding").observe(
            time.perf_counter() - wait_started
        )

    async def _compute_owned_and_local(
        self,
        local: list[_TextGroup],
        owned: list[tuple[_TextGroup, LockHandle]],
        resolved: list[list[float] | None],
    ) -> None:
        assert self._lock_store is not None
        if not local and not owned:
            return
        for _group, _handle in owned:
            SINGLE_FLIGHT_HELD_LOCKS.labels("ai", "query-embedding").inc()
        try:
            active_owned = owned
            if owned:
                reads = await self._read_many(
                    [self.key_for_canonical_text(group.text) for group, _handle in owned],
                    self._codec,
                    operation="single_flight_leader_recheck",
                )
                active_owned = []
                for (group, handle), (_outcome, vector) in zip(owned, reads, strict=True):
                    if vector is not None:
                        self._resolve_group(resolved, group, vector)
                        SINGLE_FLIGHT_EVENTS.labels(
                            "ai", "query-embedding", "joined_cache_hit"
                        ).inc()
                    else:
                        active_owned.append((group, handle))
            groups = sorted(
                [*local, *(group for group, _handle in active_owned)],
                key=lambda group: group.positions[0],
            )
            await self._compute_groups(groups, resolved)
        finally:
            for _group, handle in owned:
                try:
                    try:
                        released = await self._lock_store.release(handle)
                    except asyncio.CancelledError:
                        raise
                    except Exception:  # noqa: BLE001 - release is best effort
                        released = False
                    if not released:
                        SINGLE_FLIGHT_EVENTS.labels("ai", "query-embedding", "release_error").inc()
                finally:
                    SINGLE_FLIGHT_HELD_LOCKS.labels("ai", "query-embedding").dec()

    async def _compute_groups(
        self,
        groups: list[_TextGroup],
        resolved: list[list[float] | None],
        pending_writes: list[CacheWrite] | None = None,
    ) -> None:
        if not groups:
            return
        for group in groups:
            self._record_inputs(group.miss_outcome)
        computed = await self._wrapped.encode_texts([group.text for group in groups])
        if len(computed) != len(groups):
            raise ValueError("Text embedding encoder returned an unexpected number of vectors")
        writes: list[CacheWrite] = list(pending_writes or [])
        for group, vector in zip(groups, computed, strict=True):
            self._resolve_group(resolved, group, vector)
            if group.cacheable:
                write = self._prepare_write(group.text, vector)
                if write is not None:
                    writes.append(write)
        await self._write_many(writes)

    def key_for_text(self, text: str) -> str:
        return self.key_for_canonical_text(canonicalize_text_embedding_input(text))

    def key_for_canonical_text(self, canonical_text: str) -> str:
        return self._keys.create(
            "ai",
            "text-embedding",
            "model",
            self._model_id,
            "dim",
            str(self._dimension),
            "norm",
            TEXT_EMBEDDING_NORMALIZATION_ID,
            self._keys.sha256(canonical_text),
        )

    def legacy_key_for_text(self, text: str) -> str:
        return self.legacy_key_for_canonical_text(canonicalize_text_embedding_input(text))

    def legacy_key_for_canonical_text(self, canonical_text: str) -> str:
        return self._keys.create(
            "ai",
            "query-embedding",
            "model",
            self._model_id,
            "dim",
            str(self._dimension),
            "norm",
            TEXT_EMBEDDING_NORMALIZATION_ID,
            self._keys.sha256(canonical_text),
        )

    async def _read_many(
        self,
        keys: list[str],
        codec: TextEmbeddingBinaryCodec,
        *,
        operation: str,
    ) -> list[tuple[TextEmbeddingCacheOutcome, list[float] | None]]:
        if not keys:
            return []
        started = time.perf_counter()
        try:
            cached_values = await cache_get_many(self._cache, keys)
        finally:
            self._metrics.duration.labels(operation).observe(time.perf_counter() - started)
        return [self._decode_read(cached, codec) for cached in cached_values]

    def _decode_read(
        self,
        cached: CacheRead,
        codec: TextEmbeddingBinaryCodec,
    ) -> tuple[TextEmbeddingCacheOutcome, list[float] | None]:
        if cached.outcome is ReadOutcome.ERROR:
            return TextEmbeddingCacheOutcome.REDIS_ERROR, None
        if cached.outcome is ReadOutcome.BYPASS:
            return TextEmbeddingCacheOutcome.BYPASS, None
        if cached.outcome is ReadOutcome.MISS or cached.value is None:
            return TextEmbeddingCacheOutcome.MISS, None

        self._metrics.payload_bytes.labels("read").observe(len(cached.value))
        decoded = codec.decode(cached.value)
        if decoded.outcome is None and decoded.vector is not None:
            return TextEmbeddingCacheOutcome.HIT, decoded.vector
        return decoded.outcome or TextEmbeddingCacheOutcome.CORRUPT_DATA, None

    def _prepare_write(self, canonical_text: str, vector: list[float]) -> CacheWrite | None:
        payload = self._codec.encode(vector)
        if payload is None:
            invalid_outcome = (
                TextEmbeddingCacheOutcome.DIMENSION_MISMATCH
                if len(vector) != self._dimension
                else TextEmbeddingCacheOutcome.NON_FINITE_DATA
            )
            self._record_operations(invalid_outcome)
            return None
        self._metrics.payload_bytes.labels("write").observe(len(payload))
        return CacheWrite(
            self.key_for_canonical_text(canonical_text),
            payload,
            self._ttl_seconds,
        )

    async def _write_many(self, entries: list[CacheWrite]) -> None:
        if not entries:
            return
        started = time.perf_counter()
        try:
            write_outcomes = await cache_set_many(self._cache, entries)
        finally:
            self._metrics.duration.labels("write").observe(time.perf_counter() - started)
        for write_outcome in write_outcomes:
            if write_outcome is WriteOutcome.SUCCESS:
                self._record_operations(TextEmbeddingCacheOutcome.WRITE)
            elif write_outcome is WriteOutcome.ERROR:
                self._record_operations(TextEmbeddingCacheOutcome.REDIS_ERROR)
            else:
                self._record_operations(TextEmbeddingCacheOutcome.BYPASS)

    @staticmethod
    def _resolve_group(
        resolved: list[list[float] | None],
        group: _TextGroup,
        vector: list[float],
    ) -> None:
        for position in group.positions:
            resolved[position] = vector

    def _record_operations(self, outcome: TextEmbeddingCacheOutcome, amount: int = 1) -> None:
        self._metrics.operations.labels(outcome).inc(amount)

    def _record_inputs(self, outcome: TextEmbeddingCacheOutcome, amount: int = 1) -> None:
        self._metrics.encoder_inputs.labels(outcome).inc(amount)
