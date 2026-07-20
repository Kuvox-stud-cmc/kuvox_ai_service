from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kuvox_ai.cache import (
    CacheKeyFactory,
    CacheWrite,
    CircuitBreaker,
    DisabledCacheStore,
    JsonCacheCodec,
    ReadOutcome,
    RedisCacheStore,
    TtlJitter,
    WriteOutcome,
)
from kuvox_ai.infrastructure.redis_client import RedisClient, redact_redis_url


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def utcnow(self) -> datetime:
        return datetime(2026, 7, 16, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.now


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.fail = False
        self.delay = 0.0
        self.last_ttl: int | None = None
        self.mget_calls: list[list[str]] = []
        self.pipeline_calls = 0
        self.pipeline_responses: list[object] | None = None
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("unreachable")
        return self.values.get(key)

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("unreachable")
        self.mget_calls.append(list(keys))
        return [self.values.get(key) for key in keys]

    async def set(self, key: str, value: bytes, *, ex: int) -> bool:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("unreachable")
        self.values[key] = value
        self.last_ttl = ex
        self.ttls[key] = ex
        return True

    def pipeline(self, *, transaction: bool) -> FakePipeline:
        assert transaction is False
        self.pipeline_calls += 1
        return FakePipeline(self)

    async def delete(self, key: str) -> int:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("unreachable")
        return int(self.values.pop(key, None) is not None)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, bytes, int]] = []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    def set(self, key: str, value: bytes, *, ex: int) -> FakePipeline:
        self._commands.append((key, value, ex))
        return self

    async def execute(self, *, raise_on_error: bool) -> list[object]:
        assert raise_on_error is False
        await asyncio.sleep(self._redis.delay)
        if self._redis.fail:
            raise ConnectionError("unreachable")
        configured = self._redis.pipeline_responses
        responses = configured if configured is not None else [True] * len(self._commands)
        for (key, value, ttl), response in zip(self._commands, responses, strict=True):
            if isinstance(response, Exception):
                continue
            self._redis.values[key] = value
            self._redis.ttls[key] = ttl
        return list(responses)


@pytest.mark.asyncio
async def test_disabled_store_always_bypasses() -> None:
    store = DisabledCacheStore()
    assert (await store.get("key")).outcome is ReadOutcome.BYPASS
    assert await store.set("key", b"value", 10) is WriteOutcome.BYPASS
    assert await store.delete("key") is WriteOutcome.BYPASS


@pytest.mark.asyncio
async def test_redis_store_hit_miss_write_and_delete() -> None:
    redis = FakeRedis()
    store = RedisCacheStore(redis, jitter=TtlJitter(10, lambda low, high: high))
    assert (await store.get("key")).outcome is ReadOutcome.MISS
    assert await store.set("key", b"value", 10) is WriteOutcome.SUCCESS
    assert redis.last_ttl == 11
    hit = await store.get("key")
    assert hit == type(hit)(ReadOutcome.HIT, b"value")
    assert await store.delete("key") is WriteOutcome.SUCCESS
    assert (await store.get("key")).outcome is ReadOutcome.MISS


@pytest.mark.asyncio
async def test_redis_bulk_reads_preserve_order_and_isolate_oversized_values() -> None:
    redis = FakeRedis()
    redis.values = {"hit": b"four", "large": b"12345"}
    store = RedisCacheStore(redis, max_payload_bytes=4)

    reads = await store.get_many(["missing", "hit", "large", "hit"])

    assert reads == [
        type(reads[0])(ReadOutcome.MISS),
        type(reads[0])(ReadOutcome.HIT, b"four"),
        type(reads[0])(ReadOutcome.BYPASS),
        type(reads[0])(ReadOutcome.HIT, b"four"),
    ]
    assert redis.mget_calls == [["missing", "hit", "large", "hit"]]


@pytest.mark.asyncio
async def test_redis_bulk_writes_pipeline_independent_ttls_and_outcomes() -> None:
    redis = FakeRedis()
    jitter_values = iter((9.0, 22.0))
    store = RedisCacheStore(
        redis,
        max_payload_bytes=4,
        jitter=TtlJitter(10, lambda low, high: next(jitter_values)),
    )
    redis.pipeline_responses = [True, ConnectionError("one write failed")]

    outcomes = await store.set_many(
        [
            CacheWrite("first", b"one", 10),
            CacheWrite("large", b"12345", 10),
            CacheWrite("second", b"two", 20),
        ]
    )

    assert outcomes == [WriteOutcome.SUCCESS, WriteOutcome.BYPASS, WriteOutcome.ERROR]
    assert redis.pipeline_calls == 1
    assert redis.values == {"first": b"one"}
    assert redis.ttls == {"first": 9}


@pytest.mark.asyncio
async def test_payload_limit_bypasses_reads_and_writes() -> None:
    redis = FakeRedis()
    redis.values["large"] = b"12345"
    store = RedisCacheStore(redis, max_payload_bytes=4)
    assert (await store.get("large")).outcome is ReadOutcome.BYPASS
    assert await store.set("large", b"12345", 10) is WriteOutcome.BYPASS


def test_json_codec_validates_schema_and_required_fields() -> None:
    codec = JsonCacheCodec(FakeClock())
    encoded = codec.encode({"ok": True})
    assert codec.decode(encoded) == {"ok": True}
    assert codec.decode(b'{"schema_version":2,"created_at_utc":"x","payload":{}}') is None
    assert codec.decode(b'{"schema_version":1,"payload":{}}') is None
    assert codec.decode(b'{"schema_version":1,"created_at_utc":"x","payload":{}}') is None
    assert codec.decode(b"not-json") is None


def test_contract_fixture_matches_key_hash_and_envelope_fields() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "cache_contract.json").read_text()
    )
    factory = CacheKeyFactory(fixture["prefix"])
    assert factory.create(*fixture["parts"]) == fixture["key"]
    assert factory.sha256(fixture["canonical_sensitive_input"]) == fixture["sha256"]
    envelope = json.loads(JsonCacheCodec(FakeClock()).encode({"ok": True}))
    assert list(envelope) == fixture["envelope_fields"]


def test_ttl_jitter_is_deterministic_and_never_below_one_second() -> None:
    assert TtlJitter(10, lambda low, high: low).apply(10) == 9
    assert TtlJitter(100, lambda low, high: low).apply(1) == 1


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_returns_error() -> None:
    redis = FakeRedis()
    redis.delay = 0.1
    store = RedisCacheStore(redis, operation_timeout_seconds=0.01)
    assert (await store.get("key")).outcome is ReadOutcome.ERROR


@pytest.mark.asyncio
async def test_circuit_opens_then_allows_one_half_open_recovery() -> None:
    clock = FakeClock()
    circuit = CircuitBreaker(clock=clock, failure_threshold=5, open_seconds=10)
    redis = FakeRedis()
    redis.fail = True
    store = RedisCacheStore(redis, circuit=circuit)
    for _ in range(5):
        assert (await store.get("key")).outcome is ReadOutcome.ERROR
    assert circuit.state == "open"
    assert (await store.get("key")).outcome is ReadOutcome.BYPASS
    clock.now = 10
    redis.fail = False
    assert (await store.get("key")).outcome is ReadOutcome.MISS
    assert circuit.state == "closed"


@pytest.mark.asyncio
async def test_request_cancellation_propagates() -> None:
    redis = FakeRedis()
    redis.delay = 60
    store = RedisCacheStore(redis, operation_timeout_seconds=120)
    task = asyncio.create_task(store.get("key"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    ("url", "safe"),
    [
        ("redis://user:secret@redis:6379/0", "redis://redis:6379/0"),
        ("rediss://:secret@[::1]:6380/1?ssl=true", "rediss://[::1]:6380/1?ssl=true"),
    ],
)
def test_redis_endpoint_logging_redacts_credentials(url: str, safe: str) -> None:
    assert redact_redis_url(url) == safe
    assert "secret" not in redact_redis_url(url)


@pytest.mark.asyncio
async def test_unreachable_enabled_redis_does_not_fail_client_startup() -> None:
    client = RedisClient(
        "redis://127.0.0.1:1/0",
        enabled=True,
        connect_timeout_seconds=0.01,
        operation_timeout_seconds=0.01,
    )
    await client.connect()
    assert await client.health_check() is False
    await client.close()
