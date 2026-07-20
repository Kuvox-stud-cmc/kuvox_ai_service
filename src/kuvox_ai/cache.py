"""Shared Phase 0 cache contracts used by opt-in feature decorators."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, cast

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger
from kuvox_ai.metrics import (
    CACHE_OPERATIONS,
    CACHE_OVERSIZED_BYPASSES,
    CACHE_PAYLOAD_BYTES,
    CACHE_SCHEMA_MISSES,
    REDIS_COMMANDS,
    REDIS_LATENCY,
    set_circuit_state,
)

logger = get_logger(__name__)


class ReadOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    ERROR = "error"


class WriteOutcome(StrEnum):
    SUCCESS = "success"
    BYPASS = "bypass"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CacheRead:
    outcome: ReadOutcome
    value: bytes | None = None


@dataclass(frozen=True, slots=True)
class CacheWrite:
    key: str
    value: bytes
    ttl_seconds: int


class CacheStore(Protocol):
    async def get(self, key: str) -> CacheRead: ...

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome: ...

    async def delete(self, key: str) -> WriteOutcome: ...


class BulkCacheStore(Protocol):
    async def get_many(self, keys: list[str]) -> list[CacheRead]: ...

    async def set_many(self, entries: list[CacheWrite]) -> list[WriteOutcome]: ...


async def cache_get_many(cache: CacheStore, keys: list[str]) -> list[CacheRead]:
    """Use an optional bulk reader, falling back to independent per-key reads."""
    if not keys:
        return []
    get_many = getattr(cache, "get_many", None)
    if get_many is not None:
        try:
            reads = await cast(Callable[[list[str]], Awaitable[list[CacheRead]]], get_many)(keys)
        except Exception:  # noqa: BLE001 - optional cache must fail open
            return [CacheRead(ReadOutcome.ERROR) for _ in keys]
        if len(reads) == len(keys):
            return reads
        return [CacheRead(ReadOutcome.ERROR) for _ in keys]

    async def get_one(key: str) -> CacheRead:
        try:
            return await cache.get(key)
        except Exception:  # noqa: BLE001 - optional cache must fail open
            return CacheRead(ReadOutcome.ERROR)

    return await asyncio.gather(*(get_one(key) for key in keys))


async def cache_set_many(cache: CacheStore, entries: list[CacheWrite]) -> list[WriteOutcome]:
    """Use an optional bulk writer, falling back to independent per-key writes."""
    if not entries:
        return []
    set_many = getattr(cache, "set_many", None)
    if set_many is not None:
        try:
            outcomes = await cast(
                Callable[[list[CacheWrite]], Awaitable[list[WriteOutcome]]], set_many
            )(entries)
        except Exception:  # noqa: BLE001 - optional cache must fail open
            return [WriteOutcome.ERROR for _ in entries]
        if len(outcomes) == len(entries):
            return outcomes
        return [WriteOutcome.ERROR for _ in entries]

    async def set_one(entry: CacheWrite) -> WriteOutcome:
        try:
            return await cache.set(entry.key, entry.value, entry.ttl_seconds)
        except Exception:  # noqa: BLE001 - optional cache must fail open
            return WriteOutcome.ERROR

    return await asyncio.gather(*(set_one(entry) for entry in entries))


class Clock(Protocol):
    def utcnow(self) -> datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    def utcnow(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class TtlJitter:
    def __init__(
        self,
        percent: int = 10,
        random_source: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        self._fraction = percent / 100
        self._random_source = random_source

    def apply(self, ttl_seconds: int) -> int:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        delta = ttl_seconds * self._fraction
        return max(1, round(self._random_source(ttl_seconds - delta, ttl_seconds + delta)))


class CacheKeyFactory:
    def __init__(self, prefix: str = "kuvox:v1") -> None:
        self._prefix = prefix.strip(":")

    def create(self, *parts: str) -> str:
        normalized = [part.strip(":") for part in parts]
        if not self._prefix or not normalized or any(not part for part in normalized):
            raise ValueError("cache key prefix and parts must be non-empty")
        return ":".join((self._prefix, *normalized))

    @staticmethod
    def sha256(canonical_sensitive_input: str | bytes) -> str:
        value = (
            canonical_sensitive_input.encode("utf-8")
            if isinstance(canonical_sensitive_input, str)
            else canonical_sensitive_input
        )
        return hashlib.sha256(value).hexdigest()


class JsonCacheCodec:
    def __init__(self, clock: Clock | None = None, schema_version: int = 1) -> None:
        self._clock = clock or SystemClock()
        self._schema_version = schema_version

    def encode(self, payload: Any) -> bytes:
        envelope = {
            "schema_version": self._schema_version,
            "created_at_utc": self._clock.utcnow().isoformat().replace("+00:00", "Z"),
            "payload": payload,
        }
        return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def decode(self, value: bytes) -> Any | None:
        try:
            envelope = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(envelope, dict) or envelope.get("schema_version") != self._schema_version:
            CACHE_SCHEMA_MISSES.labels("ai").inc()
            return None
        created_at_utc = envelope.get("created_at_utc")
        if not isinstance(created_at_utc, str) or "payload" not in envelope:
            return None
        try:
            datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
        except ValueError:
            return None
        return envelope["payload"]


class DisabledCacheStore:
    async def get(self, key: str) -> CacheRead:
        del key
        CACHE_OPERATIONS.labels("ai", "get", ReadOutcome.BYPASS).inc()
        return CacheRead(ReadOutcome.BYPASS)

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        del key, value, ttl_seconds
        CACHE_OPERATIONS.labels("ai", "set", WriteOutcome.BYPASS).inc()
        return WriteOutcome.BYPASS

    async def delete(self, key: str) -> WriteOutcome:
        del key
        CACHE_OPERATIONS.labels("ai", "delete", WriteOutcome.BYPASS).inc()
        return WriteOutcome.BYPASS


class CircuitBreaker:
    def __init__(
        self,
        clock: Clock | None = None,
        failure_threshold: int = 5,
        open_seconds: float = 10,
    ) -> None:
        self._clock = clock or SystemClock()
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._failures = 0
        self._opened_at = 0.0
        self._state = "closed"
        self._half_open_in_flight = False
        set_circuit_state(self._state)

    @property
    def state(self) -> str:
        return self._state

    def allow_request(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._clock.monotonic() - self._opened_at < self._open_seconds:
                return False
            self._state = "half_open"
            set_circuit_state(self._state)
        if self._half_open_in_flight:
            return False
        self._half_open_in_flight = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._half_open_in_flight = False
        self._state = "closed"
        set_circuit_state(self._state)

    def record_failure(self) -> None:
        self._half_open_in_flight = False
        self._failures += 1
        if self._state == "half_open" or self._failures >= self._failure_threshold:
            self._state = "open"
            self._opened_at = self._clock.monotonic()
            set_circuit_state(self._state)


class RedisCacheStore:
    def __init__(
        self,
        client: Any,
        *,
        max_payload_bytes: int = 1_048_576,
        operation_timeout_seconds: float = 0.5,
        jitter: TtlJitter | None = None,
        circuit: CircuitBreaker | None = None,
    ) -> None:
        self._client = client
        self._max_payload_bytes = max_payload_bytes
        self._operation_timeout_seconds = operation_timeout_seconds
        self._jitter = jitter or TtlJitter()
        self._circuit = circuit or CircuitBreaker()

    async def get(self, key: str) -> CacheRead:
        if not self._circuit.allow_request():
            return self._read_result(ReadOutcome.BYPASS)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                value = await self._client.get(key)
            self._record_redis("get", "success", started)
            self._circuit.record_success()
            if value is None:
                return self._read_result(ReadOutcome.MISS)
            raw = bytes(value)
            if len(raw) > self._max_payload_bytes:
                CACHE_OVERSIZED_BYPASSES.labels("ai", "get").inc()
                return self._read_result(ReadOutcome.BYPASS)
            CACHE_PAYLOAD_BYTES.labels("ai", "get").observe(len(raw))
            return self._read_result(ReadOutcome.HIT, raw)
        except Exception as exc:  # noqa: BLE001 - optional cache must fail open
            self._record_failure("get", started, exc)
            return self._read_result(ReadOutcome.ERROR)

    async def get_many(self, keys: list[str]) -> list[CacheRead]:
        if not keys:
            return []
        if not self._circuit.allow_request():
            return [self._read_result(ReadOutcome.BYPASS) for _ in keys]
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                values = await self._client.mget(keys)
            if len(values) != len(keys):
                raise ValueError("Redis MGET returned an unexpected number of values")
            self._record_redis("mget", "success", started)
            self._circuit.record_success()
        except Exception as exc:  # noqa: BLE001 - optional cache must fail open
            self._record_failure("mget", started, exc)
            return [self._read_result(ReadOutcome.ERROR) for _ in keys]

        results: list[CacheRead] = []
        for value in values:
            if value is None:
                results.append(self._read_result(ReadOutcome.MISS))
                continue
            try:
                raw = bytes(value)
            except Exception:  # noqa: BLE001 - one malformed result must not poison the batch
                results.append(self._read_result(ReadOutcome.ERROR))
                continue
            if len(raw) > self._max_payload_bytes:
                CACHE_OVERSIZED_BYPASSES.labels("ai", "get").inc()
                results.append(self._read_result(ReadOutcome.BYPASS))
                continue
            CACHE_PAYLOAD_BYTES.labels("ai", "get").observe(len(raw))
            results.append(self._read_result(ReadOutcome.HIT, raw))
        return results

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        if len(value) > self._max_payload_bytes:
            CACHE_OVERSIZED_BYPASSES.labels("ai", "set").inc()
            return self._write_result("set", WriteOutcome.BYPASS)
        if not self._circuit.allow_request():
            return self._write_result("set", WriteOutcome.BYPASS)
        started = time.perf_counter()
        try:
            ttl = self._jitter.apply(ttl_seconds)
            async with asyncio.timeout(self._operation_timeout_seconds):
                await self._client.set(key, value, ex=ttl)
            self._record_redis("set", "success", started)
            self._circuit.record_success()
            CACHE_PAYLOAD_BYTES.labels("ai", "set").observe(len(value))
            return self._write_result("set", WriteOutcome.SUCCESS)
        except Exception as exc:  # noqa: BLE001 - optional cache must fail open
            self._record_failure("set", started, exc)
            return self._write_result("set", WriteOutcome.ERROR)

    async def set_many(self, entries: list[CacheWrite]) -> list[WriteOutcome]:
        if not entries:
            return []

        results: list[WriteOutcome | None] = [None] * len(entries)
        queued: list[tuple[int, CacheWrite, int]] = []
        for index, entry in enumerate(entries):
            if len(entry.value) > self._max_payload_bytes:
                CACHE_OVERSIZED_BYPASSES.labels("ai", "set").inc()
                results[index] = self._write_result("set", WriteOutcome.BYPASS)
                continue
            try:
                ttl = self._jitter.apply(entry.ttl_seconds)
            except Exception:  # noqa: BLE001 - one invalid entry must not poison the batch
                results[index] = self._write_result("set", WriteOutcome.ERROR)
                continue
            queued.append((index, entry, ttl))

        if not queued:
            return [outcome for outcome in results if outcome is not None]
        if not self._circuit.allow_request():
            for index, _entry, _ttl in queued:
                results[index] = self._write_result("set", WriteOutcome.BYPASS)
            return [outcome for outcome in results if outcome is not None]

        started = time.perf_counter()
        try:
            async with self._client.pipeline(transaction=False) as pipeline:
                for _index, entry, ttl in queued:
                    pipeline.set(entry.key, entry.value, ex=ttl)
                responses = await asyncio.wait_for(
                    pipeline.execute(raise_on_error=False),
                    timeout=self._operation_timeout_seconds,
                )
            if len(responses) != len(queued):
                raise ValueError("Redis pipeline returned an unexpected number of results")
        except Exception as exc:  # noqa: BLE001 - optional cache must fail open
            self._record_failure("pipeline_set", started, exc)
            for index, _entry, _ttl in queued:
                results[index] = self._write_result("set", WriteOutcome.ERROR)
            return [outcome for outcome in results if outcome is not None]

        failures = [response for response in responses if isinstance(response, Exception)]
        if failures:
            self._record_redis("pipeline_set", "error", started)
            self._circuit.record_failure()
            logger.warning(
                "cache.redis_failed",
                command="pipeline_set",
                error_type=type(failures[0]).__name__,
            )
        else:
            self._record_redis("pipeline_set", "success", started)
            self._circuit.record_success()

        for (index, entry, _ttl), response in zip(queued, responses, strict=True):
            if isinstance(response, Exception):
                results[index] = self._write_result("set", WriteOutcome.ERROR)
                continue
            CACHE_PAYLOAD_BYTES.labels("ai", "set").observe(len(entry.value))
            results[index] = self._write_result("set", WriteOutcome.SUCCESS)
        return [outcome for outcome in results if outcome is not None]

    async def delete(self, key: str) -> WriteOutcome:
        if not self._circuit.allow_request():
            return self._write_result("delete", WriteOutcome.BYPASS)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                await self._client.delete(key)
            self._record_redis("delete", "success", started)
            self._circuit.record_success()
            return self._write_result("delete", WriteOutcome.SUCCESS)
        except Exception as exc:  # noqa: BLE001 - optional cache must fail open
            self._record_failure("delete", started, exc)
            return self._write_result("delete", WriteOutcome.ERROR)

    def _read_result(self, outcome: ReadOutcome, value: bytes | None = None) -> CacheRead:
        CACHE_OPERATIONS.labels("ai", "get", outcome).inc()
        return CacheRead(outcome, value)

    def _write_result(self, operation: str, outcome: WriteOutcome) -> WriteOutcome:
        CACHE_OPERATIONS.labels("ai", operation, outcome).inc()
        return outcome

    def _record_redis(self, command: str, outcome: str, started: float) -> None:
        REDIS_COMMANDS.labels("ai", command, outcome).inc()
        REDIS_LATENCY.labels("ai", command).observe(time.perf_counter() - started)

    def _record_failure(self, command: str, started: float, exc: Exception) -> None:
        self._record_redis(command, "error", started)
        self._circuit.record_failure()
        logger.warning("cache.redis_failed", command=command, error_type=type(exc).__name__)


def build_cache_store(settings: Settings, client: Any) -> CacheStore:
    """Build the process-wide cache store without connecting its Redis client."""
    if not settings.cache_enabled:
        return DisabledCacheStore()
    return RedisCacheStore(
        client,
        max_payload_bytes=settings.cache_max_payload_bytes,
        operation_timeout_seconds=settings.redis_operation_timeout_seconds,
        jitter=TtlJitter(settings.cache_ttl_jitter_percent),
    )
