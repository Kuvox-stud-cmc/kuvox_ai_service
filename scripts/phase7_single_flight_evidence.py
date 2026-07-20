#!/usr/bin/env python3
"""Capture redacted Phase 7 query-embedding single-flight evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from prometheus_client import REGISTRY

from kuvox_ai.cache import CacheStore, RedisCacheStore, TtlJitter
from kuvox_ai.distributed_lock import LockStore, RedisLockStore
from kuvox_ai.infrastructure.redis_client import RedisClient
from kuvox_ai.modules.ingestion.text_encoder import (
    SentenceTransformerTextEncoder,
    TextEmbeddingEncoder,
)
from kuvox_ai.modules.retrieval.query_embedding_cache import CachedQueryTextEmbeddingEncoder


class Encoder(Protocol):
    async def encode_texts(self, texts: list[str]) -> list[list[float]]: ...


class CountingEncoder:
    def __init__(self, wrapped: Encoder) -> None:
        self._wrapped = wrapped
        self.calls = 0
        self.inputs = 0

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.inputs += len(texts)
        return await self._wrapped.encode_texts(texts)


class DeterministicEncoder:
    def __init__(self, *, delay: float = 0, fail_once: bool = False) -> None:
        self.delay = delay
        self.fail_once = fail_once
        self.calls = 0

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("redacted leader failure")
        return [[0.25, -0.5, 0.75] for _ in texts]


async def run() -> None:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    model_id = str(fixture["modelId"])
    dimension = int(fixture["dimension"])
    query_hash = hashlib.sha256(str(fixture["query"]).encode()).hexdigest()
    query_text = str(fixture["query"])
    run_id = uuid4().hex

    first_redis = RedisClient(args.redis_url)
    second_redis = RedisClient(args.redis_url)
    await first_redis.connect()
    await second_redis.connect()
    try:
        if not await first_redis.health_check():
            raise RuntimeError("real Redis is unavailable")
        await first_redis.client.flushdb()
        first_store = RedisCacheStore(first_redis, jitter=TtlJitter(0))
        second_store = RedisCacheStore(second_redis, jitter=TtlJitter(0))
        first_locks = RedisLockStore(first_redis, operation_timeout_seconds=1)
        second_locks = RedisLockStore(second_redis, operation_timeout_seconds=1)
        real = SentenceTransformerTextEncoder(
            model_name=model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
        # Load the real model once before intentionally concurrent disabled-mode calls.
        await real.encode_texts([query_text])

        disabled_source = CountingEncoder(real)
        disabled = encoder(
            disabled_source,
            first_store,
            first_locks,
            key_prefix=f"kuvox:evidence:{run_id}:disabled",
            single_flight=False,
            model_id=model_id,
            dimension=dimension,
        )
        disabled_vectors = await concurrent(disabled, query_text)

        enabled_source = CountingEncoder(real)
        enabled_one = encoder(
            enabled_source,
            first_store,
            first_locks,
            key_prefix=f"kuvox:evidence:{run_id}:enabled",
            single_flight=True,
            model_id=model_id,
            dimension=dimension,
        )
        enabled_two = encoder(
            enabled_source,
            second_store,
            second_locks,
            key_prefix=f"kuvox:evidence:{run_id}:enabled",
            single_flight=True,
            model_id=model_id,
            dimension=dimension,
        )
        enabled_vectors = await concurrent_two_instances(enabled_one, enabled_two, query_text)
        calls_after_cold = enabled_source.calls
        before_warm = command_snapshot()
        warm_vectors = await concurrent_two_instances(enabled_one, enabled_two, query_text)
        after_warm = command_snapshot()

        unavailable_first = RedisClient("redis://127.0.0.1:6399/15", connect_timeout_seconds=0.05)
        unavailable_second = RedisClient("redis://127.0.0.1:6399/15", connect_timeout_seconds=0.05)
        await unavailable_first.connect()
        await unavailable_second.connect()
        try:
            outage_source = CountingEncoder(real)
            outage_one = encoder(
                outage_source,
                RedisCacheStore(unavailable_first, operation_timeout_seconds=0.05),
                RedisLockStore(unavailable_first, operation_timeout_seconds=0.05),
                key_prefix=f"kuvox:evidence:{run_id}:outage",
                single_flight=True,
                model_id=model_id,
                dimension=dimension,
            )
            outage_two = encoder(
                outage_source,
                RedisCacheStore(unavailable_second, operation_timeout_seconds=0.05),
                RedisLockStore(unavailable_second, operation_timeout_seconds=0.05),
                key_prefix=f"kuvox:evidence:{run_id}:outage",
                single_flight=True,
                model_id=model_id,
                dimension=dimension,
            )
            outage_vectors = await concurrent_two_instances(outage_one, outage_two, query_text)
        finally:
            await unavailable_second.close()
            await unavailable_first.close()

        failure_source = DeterministicEncoder(fail_once=True)
        failure_one = encoder(
            failure_source,
            first_store,
            first_locks,
            key_prefix=f"kuvox:evidence:{run_id}:failure",
            single_flight=True,
            model_id="deterministic/phase7",
            dimension=3,
            poll_seconds=0.005,
        )
        failure_two = encoder(
            failure_source,
            second_store,
            second_locks,
            key_prefix=f"kuvox:evidence:{run_id}:failure",
            single_flight=True,
            model_id="deterministic/phase7",
            dimension=3,
            poll_seconds=0.005,
        )
        failure_results = await asyncio.gather(
            *(
                call_one(failure_one if index % 2 == 0 else failure_two, query_text)
                for index in range(16)
            ),
            return_exceptions=True,
        )
        successful_failure_vectors = [item for item in failure_results if isinstance(item, list)]

        expiry_source = DeterministicEncoder(delay=0.08)
        expiry_one = encoder(
            expiry_source,
            first_store,
            first_locks,
            key_prefix=f"kuvox:evidence:{run_id}:expiry",
            single_flight=True,
            model_id="deterministic/phase7-expiry",
            dimension=3,
            lock_ttl_seconds=0.03,
            poll_seconds=0.005,
        )
        expiry_two = encoder(
            expiry_source,
            second_store,
            second_locks,
            key_prefix=f"kuvox:evidence:{run_id}:expiry",
            single_flight=True,
            model_id="deterministic/phase7-expiry",
            dimension=3,
            lock_ttl_seconds=0.03,
            poll_seconds=0.005,
        )
        expiry_vectors = await concurrent_two_instances(expiry_one, expiry_two, query_text)

        report = {
            "schemaVersion": 1,
            "capturedAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "redis": {"kind": "real", "url": redact(args.redis_url)},
            "model": {"id": model_id, "dimension": dimension, "real": True},
            "request": {"concurrency": 16, "querySha256": query_hash},
            "singleFlightDisabled": scenario(disabled_source.calls, disabled_vectors),
            "singleFlightEnabled": scenario(enabled_source.calls, enabled_vectors),
            "warmCache": {
                **scenario(enabled_source.calls - calls_after_cold, warm_vectors),
                "redisCommandDelta": command_delta(before_warm, after_warm),
            },
            "redisUnavailable": scenario(outage_source.calls, outage_vectors),
            "leaderFailure": {
                "authoritativeCalls": failure_source.calls,
                "successfulCallers": len(successful_failure_vectors),
                "failedCallers": len(failure_results) - len(successful_failure_vectors),
                "successfulResponseSha256": response_hash(successful_failure_vectors),
            },
            "lockExpiry": scenario(expiry_source.calls, expiry_vectors),
            "twoServiceInstances": {
                "authoritativeCalls": enabled_source.calls
                - (enabled_source.calls - calls_after_cold),
                "responseSha256": response_hash(enabled_vectors),
                "identical": identical(enabled_vectors),
            },
        }
        validate(report)
        encoded = json.dumps(report, indent=2, sort_keys=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
        print(encoded)
    finally:
        if await first_redis.health_check():
            await first_redis.client.flushdb()
        await second_redis.close()
        await first_redis.close()


def encoder(
    source: Encoder,
    store: CacheStore,
    locks: LockStore,
    *,
    key_prefix: str,
    single_flight: bool,
    model_id: str,
    dimension: int,
    lock_ttl_seconds: float = 30,
    poll_seconds: float = 0.01,
) -> CachedQueryTextEmbeddingEncoder:
    return CachedQueryTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, source),
        cache=store,
        enabled=True,
        model_id=model_id,
        dimension=dimension,
        key_prefix=key_prefix,
        lock_store=locks,
        single_flight_enabled=single_flight,
        lock_ttl_seconds=lock_ttl_seconds,
        lock_wait_seconds=1,
        lock_poll_seconds=poll_seconds,
    )


async def call_one(target: CachedQueryTextEmbeddingEncoder, query: str) -> list[list[float]]:
    return await target.encode_texts([query])


async def concurrent(
    target: CachedQueryTextEmbeddingEncoder, query: str
) -> list[list[list[float]]]:
    return await asyncio.gather(*(call_one(target, query) for _ in range(16)))


async def concurrent_two_instances(
    first: CachedQueryTextEmbeddingEncoder,
    second: CachedQueryTextEmbeddingEncoder,
    query: str,
) -> list[list[list[float]]]:
    return await asyncio.gather(
        *(call_one(first if index % 2 == 0 else second, query) for index in range(16))
    )


def scenario(calls: int, vectors: list[list[list[float]]]) -> dict[str, object]:
    return {
        "authoritativeCalls": calls,
        "identical": identical(vectors),
        "responseSha256": response_hash(vectors),
    }


def identical(vectors: list[list[list[float]]]) -> bool:
    return bool(vectors) and all(item == vectors[0] for item in vectors)


def response_hash(vectors: list[list[list[float]]]) -> str | None:
    if not vectors:
        return None
    encoded = json.dumps(vectors[0], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def command_snapshot() -> dict[str, float]:
    result: dict[str, float] = {}
    for command in ("mget", "set_nx_px", "exists", "eval_release", "pipeline_set"):
        value = REGISTRY.get_sample_value(
            "kuvox_redis_commands_total",
            {"service": "ai", "command": command, "outcome": "success"},
        )
        result[command] = float(value or 0)
    return result


def command_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before if after[key] != before[key]}


def validate(report: dict[str, object]) -> None:
    disabled = cast(dict[str, object], report["singleFlightDisabled"])
    enabled = cast(dict[str, object], report["singleFlightEnabled"])
    warm = cast(dict[str, object], report["warmCache"])
    outage = cast(dict[str, object], report["redisUnavailable"])
    failure = cast(dict[str, object], report["leaderFailure"])
    expiry = cast(dict[str, object], report["lockExpiry"])
    checks = (
        int(disabled["authoritativeCalls"]) > 1,
        enabled["authoritativeCalls"] == 1,
        enabled["identical"] is True,
        warm["authoritativeCalls"] == 0,
        warm["identical"] is True,
        not any(
            key in cast(dict[str, float], warm["redisCommandDelta"])
            for key in ("set_nx_px", "exists", "eval_release")
        ),
        int(outage["authoritativeCalls"]) >= 1,
        outage["identical"] is True,
        failure["successfulCallers"] == 15,
        failure["failedCallers"] == 1,
        int(expiry["authoritativeCalls"]) > 1,
        expiry["identical"] is True,
    )
    if not all(checks):
        raise RuntimeError("Phase 7 evidence validation failed")


def redact(url: str) -> str:
    return url if "@" not in url else f"{url.split('://', 1)[0]}://{url.rsplit('@', 1)[1]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://localhost:6379/15")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/text_embedding_cache_evidence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/cache/phase7-single-flight-local.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run())
