from __future__ import annotations

import asyncio

import pytest

from kuvox_ai.distributed_lock import LockAcquire, LockAcquireOutcome, LockHandle
from kuvox_ai.single_flight import Probe, ProbeOutcome, SingleFlight


class ScriptedLocks:
    def __init__(self, outcomes: list[LockAcquireOutcome]) -> None:
        self.outcomes = outcomes
        self.locked = True
        self.release_calls = 0
        self.raise_acquire = False
        self.raise_release = False
        self.raise_is_locked = False

    async def acquire(self, component: str, cache_key: str, ttl_seconds: float) -> LockAcquire:
        del component, cache_key, ttl_seconds
        if self.raise_acquire:
            raise ConnectionError("acquire")
        outcome = self.outcomes.pop(0) if self.outcomes else LockAcquireOutcome.ACQUIRED
        return LockAcquire(outcome, LockHandle("lock:key", b"owner"))

    async def is_locked(self, handle_key: str) -> bool | None:
        del handle_key
        if self.raise_is_locked:
            raise ConnectionError("exists")
        return self.locked

    async def release(self, handle: LockHandle) -> bool:
        del handle
        self.release_calls += 1
        if self.raise_release:
            raise ConnectionError("release")
        self.locked = False
        return True


def flight(locks: ScriptedLocks, *, wait: float = 0.01) -> SingleFlight:
    return SingleFlight(
        locks,
        component="retrieval",
        lock_ttl_seconds=0.02,
        wait_seconds=wait,
        poll_seconds=0.001,
    )


@pytest.mark.asyncio
async def test_leader_failure_still_releases_owned_lock() -> None:
    locks = ScriptedLocks([LockAcquireOutcome.ACQUIRED])

    async def fail() -> str:
        raise RuntimeError("leader failed")

    with pytest.raises(RuntimeError, match="leader failed"):
        await flight(locks).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=fail,
        )

    assert locks.release_calls == 1


@pytest.mark.asyncio
async def test_release_and_acquisition_errors_do_not_fail_authoritative_work() -> None:
    release = ScriptedLocks([LockAcquireOutcome.ACQUIRED])
    release.raise_release = True
    assert (
        await flight(release).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=lambda: asyncio.sleep(0, result="value"),
        )
        == "value"
    )

    acquisition = ScriptedLocks([])
    acquisition.raise_acquire = True
    assert (
        await flight(acquisition).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=lambda: asyncio.sleep(0, result="fallback"),
        )
        == "fallback"
    )


@pytest.mark.asyncio
async def test_expired_lock_retries_leadership_and_timeout_falls_back() -> None:
    expired = ScriptedLocks([LockAcquireOutcome.CONTENDED, LockAcquireOutcome.ACQUIRED])
    expired.locked = False
    assert (
        await flight(expired).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=lambda: asyncio.sleep(0, result="after-expiry"),
        )
        == "after-expiry"
    )

    blocked = ScriptedLocks([LockAcquireOutcome.CONTENDED])
    assert (
        await flight(blocked, wait=0.002).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=lambda: asyncio.sleep(0, result="timeout-fallback"),
        )
        == "timeout-fallback"
    )


@pytest.mark.asyncio
async def test_waiter_cache_failure_and_lock_check_error_bypass_coordination() -> None:
    cache_failure = ScriptedLocks([LockAcquireOutcome.CONTENDED])
    assert (
        await flight(cache_failure).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.FAILURE)),
            authoritative=lambda: asyncio.sleep(0, result="cache-fallback"),
        )
        == "cache-fallback"
    )

    exists_failure = ScriptedLocks([LockAcquireOutcome.CONTENDED])
    exists_failure.raise_is_locked = True
    assert (
        await flight(exists_failure).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=lambda: asyncio.sleep(0, result="exists-fallback"),
        )
        == "exists-fallback"
    )


@pytest.mark.asyncio
async def test_cancellation_propagates_while_waiting() -> None:
    locks = ScriptedLocks([LockAcquireOutcome.CONTENDED])
    task = asyncio.create_task(
        flight(locks, wait=60).run(
            "key",
            probe=lambda: asyncio.sleep(0, result=Probe(ProbeOutcome.MISS)),
            authoritative=lambda: asyncio.sleep(0, result="unused"),
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
