"""Content-addressed cache decorator shared by file embedding encoders."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Generic, TypeVar

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
from kuvox_ai.embedding_cache import EmbeddingBinaryCodec, EmbeddingDecodeOutcome

FILE_HASH_CHUNK_BYTES = 1_048_576
FILE_HASH_CONCURRENCY = 4

T = TypeVar("T")


class FileEmbeddingCacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    REDIS_ERROR = "redis_error"
    CORRUPT_DATA = "corrupt_data"
    SCHEMA_MISMATCH = "schema_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    NON_FINITE_DATA = "non_finite_data"
    INPUT_CHANGED = "input_changed"
    INPUT_ERROR = "input_error"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class FileEmbeddingCacheMetrics:
    operations: Counter
    duration: Histogram
    payload_bytes: Histogram
    encoder_inputs: Counter


@dataclass(frozen=True, slots=True)
class FileSignature:
    size: int
    inode: int
    device: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    content_hash: str
    signature: FileSignature
    stable: bool


@dataclass(slots=True)
class _ContentGroup(Generic[T]):
    representative: T
    content_hash: str | None
    path_signatures: dict[Path, FileSignature]
    positions: list[int]
    stable: bool
    miss_outcome: FileEmbeddingCacheOutcome = FileEmbeddingCacheOutcome.MISS


class CachedFileEmbeddingEncoder(Generic[T]):
    """Cache vectors by exact file bytes while preserving authoritative semantics."""

    def __init__(
        self,
        encode: Callable[[list[T]], Awaitable[list[list[float]]]],
        *,
        path_of: Callable[[T], Path],
        cache: CacheStore,
        metrics: FileEmbeddingCacheMetrics,
        enabled: bool,
        namespace: str,
        magic: bytes,
        model_id: str,
        pipeline_id: str,
        dimension: int,
        ttl_seconds: int = 86_400,
        key_prefix: str = "kuvox:v1",
        hash_concurrency: int = FILE_HASH_CONCURRENCY,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if hash_concurrency <= 0:
            raise ValueError("hash_concurrency must be positive")
        self._encode = encode
        self._path_of = path_of
        self._cache = cache
        self._metrics = metrics
        self._enabled = enabled
        self._namespace = namespace
        self._dimension = dimension
        self._ttl_seconds = ttl_seconds
        self._keys = CacheKeyFactory(key_prefix)
        self._identity = self._keys.sha256(f"{model_id}\0{pipeline_id}")
        self._codec = EmbeddingBinaryCodec(
            magic=magic,
            model_id=model_id,
            pipeline_id=pipeline_id,
            dimension=dimension,
        )
        self._hash_concurrency = hash_concurrency

    async def encode_items(self, items: list[T]) -> list[list[float]]:
        if not items:
            return []
        if not self._enabled:
            self._record_operations(FileEmbeddingCacheOutcome.BYPASS, len(items))
            self._record_inputs(FileEmbeddingCacheOutcome.BYPASS, len(items))
            return await self._encode(items)

        positions_by_path: dict[Path, list[int]] = {}
        for position, item in enumerate(items):
            positions_by_path.setdefault(self._path_of(item), []).append(position)

        fingerprints = await self._fingerprint_paths(positions_by_path)
        groups = self._group_inputs(items, positions_by_path, fingerprints)
        resolved: list[list[float] | None] = [None] * len(items)
        unresolved: list[_ContentGroup[T]] = []
        stable_groups: list[_ContentGroup[T]] = []

        for group in groups:
            if not group.stable or group.content_hash is None:
                group.miss_outcome = FileEmbeddingCacheOutcome.INPUT_CHANGED
                self._record_operations(group.miss_outcome, len(group.positions))
                unresolved.append(group)
                continue
            stable_groups.append(group)

        reads = await self._read_many(
            [
                self.key_for_content_hash(group.content_hash)
                for group in stable_groups
                if group.content_hash is not None
            ]
        )
        for group, (outcome, vector) in zip(stable_groups, reads, strict=True):
            if vector is not None:
                if await self._group_is_unchanged(group):
                    self._record_operations(FileEmbeddingCacheOutcome.HIT, len(group.positions))
                    for position in group.positions:
                        resolved[position] = vector
                    continue
                outcome = FileEmbeddingCacheOutcome.INPUT_CHANGED
                group.stable = False

            group.miss_outcome = outcome
            self._record_operations(outcome, len(group.positions))
            unresolved.append(group)

        if not unresolved:
            return [vector for vector in resolved if vector is not None]

        for group in unresolved:
            self._record_inputs(group.miss_outcome)
        computed = await self._encode([group.representative for group in unresolved])
        if len(computed) != len(unresolved):
            raise ValueError("File embedding encoder returned an unexpected number of vectors")

        writes: list[CacheWrite] = []
        for group, vector in zip(unresolved, computed, strict=True):
            for position in group.positions:
                resolved[position] = vector
            if not group.stable or group.content_hash is None:
                continue
            if not await self._group_is_unchanged(group):
                self._record_operations(
                    FileEmbeddingCacheOutcome.INPUT_CHANGED,
                    len(group.positions),
                )
                continue
            write = self._prepare_write(group.content_hash, vector)
            if write is not None:
                writes.append(write)

        await self._write_many(writes)

        if any(vector is None for vector in resolved):
            raise ValueError("File embedding encoder returned an unexpected number of vectors")
        return [vector for vector in resolved if vector is not None]

    def key_for_content_hash(self, content_hash: str) -> str:
        return self._keys.create(
            "ai",
            self._namespace,
            "identity",
            self._identity,
            "dim",
            str(self._dimension),
            "content",
            content_hash,
        )

    async def _fingerprint_paths(
        self,
        positions_by_path: dict[Path, list[int]],
    ) -> dict[Path, FileFingerprint]:
        semaphore = asyncio.Semaphore(self._hash_concurrency)

        async def fingerprint(path: Path) -> tuple[Path, FileFingerprint]:
            async with semaphore:
                started = time.perf_counter()
                try:
                    value = await fingerprint_file(path)
                except Exception:
                    amount = len(positions_by_path[path])
                    self._record_operations(FileEmbeddingCacheOutcome.INPUT_ERROR, amount)
                    self._record_inputs(FileEmbeddingCacheOutcome.INPUT_ERROR, amount)
                    raise
                finally:
                    self._metrics.duration.labels("hash").observe(time.perf_counter() - started)
                return path, value

        pairs = await asyncio.gather(*(fingerprint(path) for path in positions_by_path))
        return dict(pairs)

    def _group_inputs(
        self,
        items: list[T],
        positions_by_path: dict[Path, list[int]],
        fingerprints: dict[Path, FileFingerprint],
    ) -> list[_ContentGroup[T]]:
        grouped: dict[str, _ContentGroup[T]] = {}
        for path, positions in positions_by_path.items():
            fingerprint = fingerprints[path]
            group_key = (
                f"content:{fingerprint.content_hash}" if fingerprint.stable else f"unstable:{path}"
            )
            group = grouped.get(group_key)
            if group is None:
                group = _ContentGroup(
                    representative=items[positions[0]],
                    content_hash=fingerprint.content_hash if fingerprint.stable else None,
                    path_signatures={path: fingerprint.signature},
                    positions=list(positions),
                    stable=fingerprint.stable,
                )
                grouped[group_key] = group
            else:
                group.path_signatures[path] = fingerprint.signature
                group.positions.extend(positions)
        for group in grouped.values():
            group.positions.sort()
        return sorted(grouped.values(), key=lambda group: group.positions[0])

    async def _group_is_unchanged(self, group: _ContentGroup[T]) -> bool:
        try:
            current = await asyncio.gather(
                *(_stat_signature_async(path) for path in group.path_signatures)
            )
        except Exception:
            self._record_operations(
                FileEmbeddingCacheOutcome.INPUT_ERROR,
                len(group.positions),
            )
            self._record_inputs(
                FileEmbeddingCacheOutcome.INPUT_ERROR,
                len(group.positions),
            )
            raise
        return all(
            signature == group.path_signatures[path]
            for path, signature in zip(group.path_signatures, current, strict=True)
        )

    async def _read_many(
        self,
        keys: list[str],
    ) -> list[tuple[FileEmbeddingCacheOutcome, list[float] | None]]:
        if not keys:
            return []
        started = time.perf_counter()
        try:
            cached_values = await cache_get_many(self._cache, keys)
        finally:
            self._metrics.duration.labels("read").observe(time.perf_counter() - started)
        return [self._decode_read(cached) for cached in cached_values]

    def _decode_read(
        self,
        cached: CacheRead,
    ) -> tuple[FileEmbeddingCacheOutcome, list[float] | None]:
        if cached.outcome is ReadOutcome.ERROR:
            return FileEmbeddingCacheOutcome.REDIS_ERROR, None
        if cached.outcome is ReadOutcome.BYPASS:
            return FileEmbeddingCacheOutcome.BYPASS, None
        if cached.outcome is ReadOutcome.MISS or cached.value is None:
            return FileEmbeddingCacheOutcome.MISS, None

        self._metrics.payload_bytes.labels("read").observe(len(cached.value))
        decoded = self._codec.decode(cached.value)
        if decoded.outcome is None and decoded.vector is not None:
            return FileEmbeddingCacheOutcome.HIT, decoded.vector
        if isinstance(decoded.outcome, EmbeddingDecodeOutcome):
            return FileEmbeddingCacheOutcome(decoded.outcome.value), None
        return FileEmbeddingCacheOutcome.CORRUPT_DATA, None

    def _prepare_write(self, content_hash: str, vector: list[float]) -> CacheWrite | None:
        payload = self._codec.encode(vector)
        if payload is None:
            outcome = (
                FileEmbeddingCacheOutcome.DIMENSION_MISMATCH
                if len(vector) != self._dimension
                else FileEmbeddingCacheOutcome.NON_FINITE_DATA
            )
            self._record_operations(outcome)
            return None
        self._metrics.payload_bytes.labels("write").observe(len(payload))
        return CacheWrite(
            self.key_for_content_hash(content_hash),
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
                self._record_operations(FileEmbeddingCacheOutcome.WRITE)
            elif write_outcome is WriteOutcome.ERROR:
                self._record_operations(FileEmbeddingCacheOutcome.REDIS_ERROR)
            else:
                self._record_operations(FileEmbeddingCacheOutcome.BYPASS)

    def _record_operations(self, outcome: FileEmbeddingCacheOutcome, amount: int = 1) -> None:
        self._metrics.operations.labels(outcome).inc(amount)

    def _record_inputs(self, outcome: FileEmbeddingCacheOutcome, amount: int = 1) -> None:
        self._metrics.encoder_inputs.labels(outcome).inc(amount)


async def fingerprint_file(path: Path) -> FileFingerprint:
    """Hash a file with one retry when its identity changes during the read."""
    last_digest = ""
    last_signature: FileSignature | None = None
    for _attempt in range(2):
        before = await _stat_signature_async(path)
        last_digest = await asyncio.to_thread(_sha256_file, path)
        after = await _stat_signature_async(path)
        last_signature = after
        if before == after:
            return FileFingerprint(last_digest, after, True)
    assert last_signature is not None
    return FileFingerprint(last_digest, last_signature, False)


async def _stat_signature_async(path: Path) -> FileSignature:
    return await asyncio.to_thread(_stat_signature, path)


def _stat_signature(path: Path) -> FileSignature:
    stat = path.stat()
    return FileSignature(
        size=stat.st_size,
        inode=stat.st_ino,
        device=stat.st_dev,
        mtime_ns=stat.st_mtime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(FILE_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def installed_versions(*distributions: str) -> str:
    """Return a stable dependency-version identity without importing implementations."""
    values: list[str] = []
    for distribution in distributions:
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            installed = "missing"
        values.append(f"{distribution}={installed}")
    return ",".join(values)
