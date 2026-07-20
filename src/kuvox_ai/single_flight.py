"""Reusable fail-open distributed single-flight orchestration."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from kuvox_ai.distributed_lock import LockAcquireOutcome, LockStore
from kuvox_ai.metrics import SINGLE_FLIGHT_EVENTS, SINGLE_FLIGHT_HELD_LOCKS, SINGLE_FLIGHT_WAIT

_T = TypeVar("_T")


class ProbeOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class Probe(Generic[_T]):
    outcome: ProbeOutcome
    value: _T | None = None


class SingleFlight:
    def __init__(
        self,
        locks: LockStore,
        *,
        component: str,
        lock_ttl_seconds: float,
        wait_seconds: float,
        poll_seconds: float,
    ) -> None:
        self._locks = locks
        self._component = component
        self._lock_ttl_seconds = lock_ttl_seconds
        self._wait_seconds = wait_seconds
        self._poll_seconds = poll_seconds

    async def run(
        self,
        cache_key: str,
        *,
        probe: Callable[[], Awaitable[Probe[_T]]],
        authoritative: Callable[[], Awaitable[_T]],
    ) -> _T:
        deadline = time.monotonic() + self._wait_seconds
        while True:
            try:
                attempt = await self._locks.acquire(
                    self._component,
                    cache_key,
                    self._lock_ttl_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - optional coordination fails open
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "acquisition_error").inc()
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "authoritative_fallback").inc()
                return await authoritative()
            if attempt.outcome is LockAcquireOutcome.ACQUIRED and attempt.handle is not None:
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "leader").inc()
                SINGLE_FLIGHT_HELD_LOCKS.labels("ai", self._component).inc()
                try:
                    cached = await probe()
                    if cached.outcome is ProbeOutcome.HIT and cached.value is not None:
                        SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "joined_cache_hit").inc()
                        return cached.value
                    return await authoritative()
                finally:
                    try:
                        try:
                            released = await self._locks.release(attempt.handle)
                        except asyncio.CancelledError:
                            raise
                        except Exception:  # noqa: BLE001 - release is best effort
                            released = False
                        if not released:
                            SINGLE_FLIGHT_EVENTS.labels(
                                "ai", self._component, "release_error"
                            ).inc()
                    finally:
                        SINGLE_FLIGHT_HELD_LOCKS.labels("ai", self._component).dec()

            if attempt.outcome in {LockAcquireOutcome.BYPASS, LockAcquireOutcome.ERROR}:
                outcome = (
                    "bypass"
                    if attempt.outcome is LockAcquireOutcome.BYPASS
                    else "acquisition_error"
                )
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, outcome).inc()
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "authoritative_fallback").inc()
                return await authoritative()

            SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "join").inc()
            wait_started = time.perf_counter()
            handle = attempt.handle
            while time.monotonic() < deadline:
                await asyncio.sleep(self._jittered_poll())
                cached = await probe()
                if cached.outcome is ProbeOutcome.HIT and cached.value is not None:
                    SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "joined_cache_hit").inc()
                    SINGLE_FLIGHT_WAIT.labels("ai", self._component).observe(
                        time.perf_counter() - wait_started
                    )
                    return cached.value
                if cached.outcome is ProbeOutcome.FAILURE:
                    SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "bypass").inc()
                    SINGLE_FLIGHT_EVENTS.labels(
                        "ai", self._component, "authoritative_fallback"
                    ).inc()
                    SINGLE_FLIGHT_WAIT.labels("ai", self._component).observe(
                        time.perf_counter() - wait_started
                    )
                    return await authoritative()
                if handle is None:
                    break
                try:
                    locked = await self._locks.is_locked(handle.key)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - optional coordination fails open
                    locked = None
                if locked is None:
                    SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "bypass").inc()
                    SINGLE_FLIGHT_EVENTS.labels(
                        "ai", self._component, "authoritative_fallback"
                    ).inc()
                    SINGLE_FLIGHT_WAIT.labels("ai", self._component).observe(
                        time.perf_counter() - wait_started
                    )
                    return await authoritative()
                if not locked:
                    break
            else:
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "timeout").inc()
                SINGLE_FLIGHT_EVENTS.labels("ai", self._component, "authoritative_fallback").inc()
                SINGLE_FLIGHT_WAIT.labels("ai", self._component).observe(
                    time.perf_counter() - wait_started
                )
                return await authoritative()

            SINGLE_FLIGHT_WAIT.labels("ai", self._component).observe(
                time.perf_counter() - wait_started
            )

    def _jittered_poll(self) -> float:
        return max(0.001, self._poll_seconds * random.uniform(0.8, 1.2))
