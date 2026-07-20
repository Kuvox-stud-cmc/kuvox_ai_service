"""Optional Redis-backed advisory locks for cache single-flight coordination."""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from kuvox_ai.cache import CacheKeyFactory, CircuitBreaker
from kuvox_ai.logging import get_logger
from kuvox_ai.metrics import REDIS_COMMANDS, REDIS_LATENCY, SINGLE_FLIGHT_EVENTS

logger = get_logger(__name__)

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


class LockAcquireOutcome(StrEnum):
    ACQUIRED = "acquired"
    CONTENDED = "contended"
    BYPASS = "bypass"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LockHandle:
    key: str
    owner: bytes


@dataclass(frozen=True, slots=True)
class LockAcquire:
    outcome: LockAcquireOutcome
    handle: LockHandle | None = None


class LockStore(Protocol):
    async def acquire(self, component: str, cache_key: str, ttl_seconds: float) -> LockAcquire: ...

    async def is_locked(self, handle_key: str) -> bool | None: ...

    async def release(self, handle: LockHandle) -> bool: ...


class DisabledLockStore:
    async def acquire(self, component: str, cache_key: str, ttl_seconds: float) -> LockAcquire:
        del component, cache_key, ttl_seconds
        return LockAcquire(LockAcquireOutcome.BYPASS)

    async def is_locked(self, handle_key: str) -> bool | None:
        del handle_key
        return None

    async def release(self, handle: LockHandle) -> bool:
        del handle
        return False


class RedisLockStore:
    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "kuvox:v1",
        operation_timeout_seconds: float = 0.5,
        circuit: CircuitBreaker | None = None,
    ) -> None:
        self._client = client
        self._keys = CacheKeyFactory(key_prefix)
        self._operation_timeout_seconds = operation_timeout_seconds
        self._circuit = circuit or CircuitBreaker()

    def key_for(self, component: str, cache_key: str) -> str:
        return self._keys.create("ai", "lock", component, self._keys.sha256(cache_key))

    async def acquire(self, component: str, cache_key: str, ttl_seconds: float) -> LockAcquire:
        if not self._circuit.allow_request():
            return LockAcquire(LockAcquireOutcome.BYPASS)
        lock_key = self.key_for(component, cache_key)
        owner = secrets.token_hex(16).encode("ascii")
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                acquired = await self._client.set_if_absent(
                    lock_key,
                    owner,
                    ttl_milliseconds=max(1, round(ttl_seconds * 1000)),
                )
            self._record("set_nx_px", "success", started)
            self._circuit.record_success()
            if acquired:
                return LockAcquire(LockAcquireOutcome.ACQUIRED, LockHandle(lock_key, owner))
            return LockAcquire(LockAcquireOutcome.CONTENDED, LockHandle(lock_key, owner))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional locks fail open
            self._failure("set_nx_px", started, exc)
            return LockAcquire(LockAcquireOutcome.ERROR)

    async def is_locked(self, handle_key: str) -> bool | None:
        if not self._circuit.allow_request():
            return None
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                exists = await self._client.exists(handle_key)
            self._record("exists", "success", started)
            self._circuit.record_success()
            return bool(exists)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._failure("exists", started, exc)
            return None

    async def release(self, handle: LockHandle) -> bool:
        if not self._circuit.allow_request():
            return False
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._operation_timeout_seconds):
                deleted = await self._client.eval(_RELEASE_SCRIPT, [handle.key], [handle.owner])
            self._record("eval_release", "success", started)
            self._circuit.record_success()
            return int(deleted) == 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._failure("eval_release", started, exc)
            return False

    @staticmethod
    def _record(command: str, outcome: str, started: float) -> None:
        REDIS_COMMANDS.labels("ai", command, outcome).inc()
        REDIS_LATENCY.labels("ai", command).observe(time.perf_counter() - started)

    def _failure(self, command: str, started: float, exc: Exception) -> None:
        self._record(command, "error", started)
        self._circuit.record_failure()
        SINGLE_FLIGHT_EVENTS.labels("ai", "lock", "acquisition_error").inc()
        logger.warning("single_flight.redis_failed", command=command, error_type=type(exc).__name__)


def build_lock_store(settings: Any, client: Any) -> LockStore:
    if not settings.cache_enabled:
        return DisabledLockStore()
    return RedisLockStore(
        client,
        key_prefix=settings.cache_key_prefix,
        operation_timeout_seconds=settings.redis_operation_timeout_seconds,
    )
