#!/usr/bin/env python3
"""Capture local production-like evidence for shared text embedding caching."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Protocol, cast
from uuid import uuid4

import redis.asyncio as aioredis
from prometheus_client import REGISTRY

from kuvox_ai.cache import CacheRead, CacheStore, RedisCacheStore, TtlJitter, WriteOutcome
from kuvox_ai.infrastructure import KuzuClient, QdrantClient
from kuvox_ai.modules.ingestion.text_embedding_cache import CachedIngestionTextEmbeddingEncoder
from kuvox_ai.modules.ingestion.text_encoder import (
    SentenceTransformerTextEncoder,
    TextEmbeddingEncoder,
)
from kuvox_ai.modules.retrieval.models import VideoEditorRetrievalQuery
from kuvox_ai.modules.retrieval.query_embedding_cache import (
    CachedQueryTextEmbeddingEncoder,
    QueryEmbeddingBinaryCodec,
)
from kuvox_ai.modules.retrieval.service import RetrievalService

_COUNTERS = (
    "kuvox_query_embedding_cache_operations_total",
    "kuvox_query_embedding_encoder_inputs_total",
    "kuvox_ingestion_text_embedding_cache_operations_total",
    "kuvox_ingestion_text_embedding_encoder_inputs_total",
)


class Encoder(Protocol):
    async def encode_texts(self, texts: list[str]) -> list[list[float]]: ...


class CountingEncoder:
    def __init__(self, wrapped: Encoder) -> None:
        self._wrapped = wrapped
        self.inputs = 0
        self.calls = 0

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.inputs += len(texts)
        return await self._wrapped.encode_texts(texts)


class UnavailableCacheStore:
    async def get(self, key: str) -> CacheRead:
        del key
        raise ConnectionError("simulated Redis outage")


class EvidenceQdrant:
    def __init__(self) -> None:
        self.client = self

    async def collection_exists(self, collection_name: str) -> bool:
        del collection_name
        return True

    async def query_points(self, **kwargs: Any) -> Any:
        del kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.91,
                    payload={
                        "shotId": "evidence-media:shot:000001",
                        "mediaId": "evidence-media",
                        "startSeconds": 1.0,
                        "endSeconds": 4.0,
                        "text": "anonymized retrieval evidence",
                    },
                )
            ]
        )


class EvidenceKuzu:
    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        del query, params
        return []

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        del key, value, ttl_seconds
        raise ConnectionError("simulated Redis outage")

    async def delete(self, key: str) -> WriteOutcome:
        del key
        raise ConnectionError("simulated Redis outage")


async def run() -> None:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    validate_fixture(fixture)
    model_id = str(fixture["modelId"])
    dimension = int(fixture["dimension"])
    query_text = str(fixture["query"])
    legacy_text = str(fixture["legacyText"])
    ingestion_texts = flatten_ingestion_texts(cast(dict[str, Any], fixture["ingestionTexts"]))
    run_id = uuid4().hex
    key_prefix = f"kuvox:evidence:{run_id}"

    redis = aioredis.from_url(
        args.redis_url,
        decode_responses=False,
        socket_connect_timeout=args.redis_timeout_seconds,
        socket_timeout=args.redis_timeout_seconds,
    )
    keys_to_delete: set[str] = set()
    try:
        await redis.ping()
        store = RedisCacheStore(redis, jitter=TtlJitter(0))
        real_encoder = SentenceTransformerTextEncoder(
            model_name=model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
        disabled_counter = CountingEncoder(real_encoder)
        disabled = query_encoder(
            disabled_counter,
            cast(CacheStore, store),
            enabled=False,
            model_id=model_id,
            dimension=dimension,
            key_prefix=key_prefix,
        )
        disabled_first_started = perf_counter()
        disabled_first = await disabled.encode_texts([query_text])
        disabled_first_ms = elapsed_ms(disabled_first_started)
        disabled_second_started = perf_counter()
        disabled_second = await disabled.encode_texts([query_text])
        disabled_second_ms = elapsed_ms(disabled_second_started)

        query_counter = CountingEncoder(real_encoder)
        query = query_encoder(
            query_counter,
            cast(CacheStore, store),
            enabled=True,
            model_id=model_id,
            dimension=dimension,
            key_prefix=key_prefix,
        )
        ingestion_counter = CountingEncoder(real_encoder)
        ingestion = ingestion_encoder(
            ingestion_counter,
            cast(CacheStore, store),
            enabled=True,
            model_id=model_id,
            dimension=dimension,
            key_prefix=key_prefix,
        )
        before = metric_snapshot()
        commands_before = redis_command_snapshot()
        query_warm_started = perf_counter()
        query_warm = await query.encode_texts([query_text])
        query_warm_ms = elapsed_ms(query_warm_started)
        after_query_warm = metric_snapshot()
        commands_after_query_warm = redis_command_snapshot()
        query_warm_inputs = query_counter.inputs
        query_repeat_started = perf_counter()
        query_repeat = await query.encode_texts([query_text])
        query_repeat_ms = elapsed_ms(query_repeat_started)
        after_query_repeat = metric_snapshot()
        commands_after_query_repeat = redis_command_snapshot()
        query_repeat_inputs = query_counter.inputs - query_warm_inputs
        ingestion_warm_started = perf_counter()
        ingestion_warm = await ingestion.encode_texts(ingestion_texts)
        ingestion_warm_ms = elapsed_ms(ingestion_warm_started)
        after_ingestion_warm = metric_snapshot()
        commands_after_ingestion_warm = redis_command_snapshot()
        ingestion_warm_inputs = ingestion_counter.inputs
        ingestion_repeat_started = perf_counter()
        ingestion_repeat = await ingestion.encode_texts(ingestion_texts)
        ingestion_repeat_ms = elapsed_ms(ingestion_repeat_started)
        after_ingestion_repeat = metric_snapshot()
        commands_after_ingestion_repeat = redis_command_snapshot()
        ingestion_repeat_inputs = ingestion_counter.inputs - ingestion_warm_inputs

        legacy_vector = (await real_encoder.encode_texts([legacy_text]))[0]
        legacy_payload = QueryEmbeddingBinaryCodec(
            model_id=model_id,
            dimension=dimension,
        ).encode(legacy_vector)
        if legacy_payload is None:
            raise RuntimeError("real encoder produced a non-cacheable legacy vector")
        legacy_key = query.legacy_key_for_text(legacy_text)
        shared_legacy_key = query.key_for_text(legacy_text)
        keys_to_delete.update((legacy_key, shared_legacy_key))
        await redis.set(legacy_key, legacy_payload, ex=60)
        legacy_inputs_before = query_counter.inputs
        promoted = await query.encode_texts([legacy_text])
        promoted_raw = await redis.get(shared_legacy_key)

        outage_counter = CountingEncoder(real_encoder)
        outage = query_encoder(
            outage_counter,
            cast(CacheStore, UnavailableCacheStore()),
            enabled=True,
            model_id=model_id,
            dimension=dimension,
            key_prefix=f"{key_prefix}:outage",
        )
        outage_started = perf_counter()
        outage_result = await outage.encode_texts([query_text])
        outage_ms = elapsed_ms(outage_started)

        retrieval_counter = CountingEncoder(real_encoder)
        retrieval_encoder = query_encoder(
            retrieval_counter,
            cast(CacheStore, store),
            enabled=True,
            model_id=model_id,
            dimension=dimension,
            key_prefix=f"{key_prefix}:retrieval",
        )
        retrieval = RetrievalService(
            kuzu=cast(KuzuClient, EvidenceKuzu()),
            qdrant=cast(QdrantClient, EvidenceQdrant()),
            text_encoder=retrieval_encoder,
        )
        retrieval_request = VideoEditorRetrievalQuery(
            projectId="evidence-project",
            mediaIds=["evidence-media"],
            query=query_text,
            modalities=["transcript"],
            expandGraph=False,
        )
        retrieval_warm = await retrieval.retrieve_video_editor(retrieval_request)
        retrieval_warm_inputs = retrieval_counter.inputs
        retrieval_repeat = await retrieval.retrieve_video_editor(retrieval_request)
        retrieval_repeat_inputs = retrieval_counter.inputs - retrieval_warm_inputs

        keys_to_delete.update(query.key_for_text(text) for text in [query_text, *ingestion_texts])
        keys_to_delete.add(retrieval_encoder.key_for_text(query_text))
        redis_stats = await redis_namespace_stats(redis, key_prefix)
        report = {
            "schemaVersion": 1,
            "capturedAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "runId": run_id,
            "fixtureSha256": sha256_json(fixture),
            "modelId": model_id,
            "dimension": dimension,
            "redisUrl": redact_redis_url(args.redis_url),
            "disabled": {
                "encoderInputs": disabled_counter.inputs,
                "equivalent": vectors_equal(disabled_first, disabled_second),
                "vectorSha256": sha256_json(disabled_first),
                "firstDurationMs": disabled_first_ms,
                "repeatDurationMs": disabled_second_ms,
            },
            "query": {
                "warmEncoderInputs": query_warm_inputs,
                "repeatEncoderInputs": query_repeat_inputs,
                "warmEquivalentToDisabled": vectors_equal(query_warm, disabled_first),
                "repeatEquivalent": vectors_equal(query_repeat, query_warm),
                "warmDurationMs": query_warm_ms,
                "repeatDurationMs": query_repeat_ms,
                "vectorSha256": sha256_json(query_warm),
                "warmMetricDelta": metric_delta(before, after_query_warm),
                "repeatMetricDelta": metric_delta(after_query_warm, after_query_repeat),
                "warmRedisCommandDelta": redis_command_delta(
                    commands_before, commands_after_query_warm
                ),
                "repeatRedisCommandDelta": redis_command_delta(
                    commands_after_query_warm, commands_after_query_repeat
                ),
            },
            "ingestion": {
                "inputCount": len(ingestion_texts),
                "warmEncoderInputs": ingestion_warm_inputs,
                "repeatEncoderInputs": ingestion_repeat_inputs,
                "repeatEquivalent": vectors_equal(ingestion_repeat, ingestion_warm),
                "warmDurationMs": ingestion_warm_ms,
                "repeatDurationMs": ingestion_repeat_ms,
                "vectorSha256": sha256_json(ingestion_warm),
                "warmMetricDelta": metric_delta(after_query_repeat, after_ingestion_warm),
                "repeatMetricDelta": metric_delta(after_ingestion_warm, after_ingestion_repeat),
                "warmRedisCommandDelta": redis_command_delta(
                    commands_after_query_repeat, commands_after_ingestion_warm
                ),
                "repeatRedisCommandDelta": redis_command_delta(
                    commands_after_ingestion_warm, commands_after_ingestion_repeat
                ),
            },
            "crossConsumerReuse": {
                "sharedQueryTextPresentInIngestion": query_text in ingestion_texts,
                "ingestionWarmInputsSaved": len(ingestion_texts) - ingestion_warm_inputs,
            },
            "legacyPromotion": {
                "encoderInputs": query_counter.inputs - legacy_inputs_before,
                "equivalent": vectors_equal(promoted, [legacy_vector]),
                "sharedValueMagic": (
                    bytes(promoted_raw[:4]).decode("ascii")
                    if isinstance(promoted_raw, bytes)
                    else None
                ),
            },
            "redisOutage": {
                "encoderInputs": outage_counter.inputs,
                "equivalentToDisabled": vectors_equal(outage_result, disabled_first),
                "durationMs": outage_ms,
            },
            "retrievalDto": {
                "warmEncoderInputs": retrieval_warm_inputs,
                "repeatEncoderInputs": retrieval_repeat_inputs,
                "equivalent": retrieval_warm == retrieval_repeat,
                "responseSha256": sha256_json(
                    retrieval_warm.model_dump(by_alias=True, mode="json")
                ),
            },
            "redis": redis_stats,
        }
        validate_report(report)
        encoded = json.dumps(report, indent=2, sort_keys=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
        print(encoded)
    finally:
        if keys_to_delete:
            await redis.delete(*keys_to_delete)
        await redis.aclose()


def query_encoder(
    encoder: CountingEncoder,
    cache: CacheStore,
    *,
    enabled: bool,
    model_id: str,
    dimension: int,
    key_prefix: str,
) -> CachedQueryTextEmbeddingEncoder:
    return CachedQueryTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, encoder),
        cache=cache,
        enabled=enabled,
        model_id=model_id,
        dimension=dimension,
        key_prefix=key_prefix,
        legacy_read_enabled=True,
    )


def ingestion_encoder(
    encoder: CountingEncoder,
    cache: CacheStore,
    *,
    enabled: bool,
    model_id: str,
    dimension: int,
    key_prefix: str,
) -> CachedIngestionTextEmbeddingEncoder:
    return CachedIngestionTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, encoder),
        cache=cache,
        enabled=enabled,
        model_id=model_id,
        dimension=dimension,
        key_prefix=key_prefix,
        legacy_read_enabled=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/text_embedding_cache_evidence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/cache/phase1-phase2-local.json"),
    )
    parser.add_argument("--redis-url", default="redis://localhost:6379/15")
    parser.add_argument("--redis-timeout-seconds", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def validate_fixture(fixture: Any) -> None:
    if not isinstance(fixture, dict) or fixture.get("schemaVersion") != 1:
        raise ValueError("unsupported evidence fixture")
    for field in ("modelId", "dimension", "query", "legacyText", "ingestionTexts"):
        if field not in fixture:
            raise ValueError(f"evidence fixture is missing {field}")


def flatten_ingestion_texts(groups: dict[str, Any]) -> list[str]:
    expected = ("shotTranscripts", "shotOcr", "mediaTranscripts", "mediaOcr")
    texts: list[str] = []
    for group in expected:
        values = groups.get(group)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"ingestion text group {group} must be a list of strings")
        texts.extend(values)
    return texts


def metric_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    outcomes = (
        "hit",
        "legacy_hit",
        "miss",
        "bypass",
        "redis_error",
        "corrupt_data",
        "schema_mismatch",
        "identity_mismatch",
        "dimension_mismatch",
        "non_finite_data",
        "write",
    )
    for name in _COUNTERS:
        for outcome in outcomes:
            value = REGISTRY.get_sample_value(name, {"outcome": outcome})
            snapshot[f'{name}{{outcome="{outcome}"}}'] = float(value or 0)
    return snapshot


def metric_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(before.keys() | after.keys())
        if after.get(key, 0) != before.get(key, 0)
    }


def redis_command_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for command in ("get", "set", "mget", "pipeline_set", "delete"):
        for outcome in ("success", "error"):
            value = REGISTRY.get_sample_value(
                "kuvox_redis_commands_total",
                {"service": "ai", "command": command, "outcome": outcome},
            )
            snapshot[f"{command}:{outcome}"] = float(value or 0)
    return snapshot


def redis_command_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(before.keys() | after.keys())
        if after.get(key, 0) != before.get(key, 0)
    }


async def redis_namespace_stats(redis: Any, key_prefix: str) -> dict[str, Any]:
    keys = [key async for key in redis.scan_iter(f"{key_prefix}:*")]
    payload_sizes = [int(await redis.strlen(key)) for key in keys]
    memory_sizes = [int(await redis.memory_usage(key) or 0) for key in keys]
    return {
        "keyCardinality": len(keys),
        "payloadBytes": {
            "minimum": min(payload_sizes, default=0),
            "maximum": max(payload_sizes, default=0),
            "total": sum(payload_sizes),
        },
        "memoryBytes": sum(memory_sizes),
    }


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def vectors_equal(left: list[list[float]], right: list[list[float]]) -> bool:
    return sha256_json(left) == sha256_json(right)


def validate_report(report: dict[str, Any]) -> None:
    checks = (
        report["disabled"]["encoderInputs"] == 2,
        report["disabled"]["equivalent"],
        report["query"]["warmEquivalentToDisabled"],
        report["query"]["repeatEquivalent"],
        report["query"]["repeatEncoderInputs"] == 0,
        report["ingestion"]["repeatEquivalent"],
        report["ingestion"]["repeatEncoderInputs"] == 0,
        report["crossConsumerReuse"]["ingestionWarmInputsSaved"] >= 1,
        report["legacyPromotion"]["encoderInputs"] == 0,
        report["legacyPromotion"]["equivalent"],
        report["legacyPromotion"]["sharedValueMagic"] == "KTEV",
        report["redisOutage"]["encoderInputs"] == 1,
        report["redisOutage"]["equivalentToDisabled"],
        report["retrievalDto"]["warmEncoderInputs"] == 1,
        report["retrievalDto"]["repeatEncoderInputs"] == 0,
        report["retrievalDto"]["equivalent"],
        report["query"]["warmRedisCommandDelta"]
        == {"mget:success": 2.0, "pipeline_set:success": 1.0},
        report["query"]["repeatRedisCommandDelta"] == {"mget:success": 1.0},
        report["ingestion"]["warmRedisCommandDelta"]
        == {"mget:success": 2.0, "pipeline_set:success": 1.0},
        report["ingestion"]["repeatRedisCommandDelta"] == {"mget:success": 1.0},
        report["redis"]["keyCardinality"] > 0,
        report["redis"]["payloadBytes"]["total"] > 0,
        report["redis"]["memoryBytes"] > 0,
    )
    if not all(checks):
        raise RuntimeError("text embedding cache evidence validation failed")


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def redact_redis_url(url: str) -> str:
    if "@" not in url:
        return url
    return f"{url.split('://', 1)[0]}://{url.rsplit('@', 1)[1]}"


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
