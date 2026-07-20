from __future__ import annotations

import asyncio

import pytest

from kuvox_ai.cache import CircuitBreaker
from kuvox_ai.distributed_lock import LockAcquireOutcome, RedisLockStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False
        self.delay = 0.0
        self.fail_release = False

    async def set_if_absent(self, key: str, value: bytes, *, ttl_milliseconds: int) -> bool:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("offline")
        if key in self.values:
            return False
        self.values[key] = value
        self.ttls[key] = ttl_milliseconds
        return True

    async def exists(self, key: str) -> bool:
        if self.fail:
            raise ConnectionError("offline")
        return key in self.values

    async def eval(self, script: str, keys: list[str], args: list[bytes]) -> int:
        del script
        if self.fail_release:
            raise ConnectionError("release offline")
        key = keys[0]
        if self.values.get(key) != args[0]:
            return 0
        self.values.pop(key, None)
        return 1


@pytest.mark.asyncio
async def test_set_nx_px_contention_and_owner_safe_release() -> None:
    redis = FakeRedis()
    store = RedisLockStore(redis, operation_timeout_seconds=1)

    first = await store.acquire("retrieval", "private-cache-key", 30)
    second = await store.acquire("retrieval", "private-cache-key", 30)

    assert first.outcome is LockAcquireOutcome.ACQUIRED
    assert second.outcome is LockAcquireOutcome.CONTENDED
    assert first.handle is not None and second.handle is not None
    assert first.handle.key.startswith("kuvox:v1:ai:lock:retrieval:")
    assert "private-cache-key" not in first.handle.key
    assert redis.ttls[first.handle.key] == 30_000
    assert await store.release(second.handle) is False
    assert await store.is_locked(first.handle.key) is True
    assert await store.release(first.handle) is True
    assert await store.is_locked(first.handle.key) is False


@pytest.mark.asyncio
async def test_lock_errors_fail_open() -> None:
    redis = FakeRedis()
    redis.fail = True
    store = RedisLockStore(redis, operation_timeout_seconds=1)

    assert (await store.acquire("retrieval", "key", 30)).outcome is LockAcquireOutcome.ERROR


@pytest.mark.asyncio
async def test_timeout_circuit_bypass_and_release_errors_fail_open() -> None:
    slow = FakeRedis()
    slow.delay = 0.05
    timed = RedisLockStore(slow, operation_timeout_seconds=0.001)
    assert (await timed.acquire("retrieval", "key", 30)).outcome is LockAcquireOutcome.ERROR

    offline = FakeRedis()
    offline.fail = True
    circuit = CircuitBreaker(failure_threshold=2, open_seconds=60)
    store = RedisLockStore(offline, operation_timeout_seconds=1, circuit=circuit)
    assert (await store.acquire("retrieval", "one", 30)).outcome is LockAcquireOutcome.ERROR
    assert (await store.acquire("retrieval", "two", 30)).outcome is LockAcquireOutcome.ERROR
    assert (await store.acquire("retrieval", "three", 30)).outcome is LockAcquireOutcome.BYPASS

    release_redis = FakeRedis()
    release_store = RedisLockStore(release_redis, operation_timeout_seconds=1)
    acquired = await release_store.acquire("retrieval", "release", 30)
    assert acquired.handle is not None
    release_redis.fail_release = True
    assert await release_store.release(acquired.handle) is False


@pytest.mark.asyncio
async def test_explicit_cancellation_propagates() -> None:
    redis = FakeRedis()
    redis.delay = 60
    store = RedisLockStore(redis, operation_timeout_seconds=120)
    task = asyncio.create_task(store.acquire("retrieval", "cancel", 30))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
