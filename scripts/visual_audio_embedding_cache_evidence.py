#!/usr/bin/env python3
"""Capture redacted real-model evidence for Phase 3 file embedding caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast
from uuid import uuid4

import redis.asyncio as aioredis
from PIL import Image
from prometheus_client import REGISTRY

from kuvox_ai.cache import CacheRead, CacheStore, RedisCacheStore, TtlJitter, WriteOutcome
from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.audio_embedding_cache import (
    AUDIO_EMBEDDING_PIPELINE_ID,
    CachedAudioEmbeddingEncoder,
    audio_model_id,
)
from kuvox_ai.modules.ingestion.audio_encoder import AudioEmbeddingEncoder, MsClapAudioEncoder
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.visual_embedding_cache import (
    VISUAL_EMBEDDING_PIPELINE_ID,
    CachedVisualEmbeddingEncoder,
    visual_model_id,
)
from kuvox_ai.modules.ingestion.visual_encoder import ClipVisualEncoder, VisualEncoder

_COUNTERS = (
    "kuvox_visual_embedding_cache_operations_total",
    "kuvox_visual_embedding_encoder_inputs_total",
    "kuvox_audio_embedding_cache_operations_total",
    "kuvox_audio_embedding_encoder_inputs_total",
)
_OUTCOMES = (
    "hit",
    "miss",
    "bypass",
    "redis_error",
    "corrupt_data",
    "schema_mismatch",
    "identity_mismatch",
    "dimension_mismatch",
    "non_finite_data",
    "input_changed",
    "input_error",
    "write",
)


class VisualProtocol(Protocol):
    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]: ...


class AudioProtocol(Protocol):
    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]: ...


class CountingVisualEncoder:
    def __init__(self, wrapped: VisualProtocol) -> None:
        self._wrapped = wrapped
        self.calls = 0
        self.inputs = 0

    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
        self.calls += 1
        self.inputs += len(frames)
        return await self._wrapped.encode_frames(frames)


class CountingAudioEncoder:
    def __init__(self, wrapped: AudioProtocol) -> None:
        self._wrapped = wrapped
        self.calls = 0
        self.inputs = 0

    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        self.calls += 1
        self.inputs += len(clips)
        return await self._wrapped.encode_audio_clips(clips)


class UnavailableCacheStore:
    async def get(self, key: str) -> CacheRead:
        del key
        raise ConnectionError("simulated Redis outage")

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        del key, value, ttl_seconds
        raise ConnectionError("simulated Redis outage")

    async def delete(self, key: str) -> WriteOutcome:
        del key
        raise ConnectionError("simulated Redis outage")


async def run() -> None:
    args = parse_args()
    key_prefix = f"kuvox:evidence:phase3:{uuid4().hex}"
    redis = aioredis.from_url(
        args.redis_url,
        decode_responses=False,
        socket_connect_timeout=args.redis_timeout_seconds,
        socket_timeout=args.redis_timeout_seconds,
    )
    try:
        await redis.ping()
        store = RedisCacheStore(redis, jitter=TtlJitter(0))
        with tempfile.TemporaryDirectory(prefix="kuvox-phase3-") as directory:
            fixture_dir = Path(directory)
            fixtures = create_fixtures(fixture_dir)
            report = {
                "schemaVersion": 1,
                "redisUrl": redact_redis_url(args.redis_url),
                "modelIdentities": {
                    "visual": visual_model_id(
                        args.clip_model_name,
                        args.clip_pretrained,
                        512,
                    ),
                    "audio": audio_model_id(1024),
                },
                "pipelineIdentities": {
                    "visual": VISUAL_EMBEDDING_PIPELINE_ID,
                    "audio": AUDIO_EMBEDDING_PIPELINE_ID,
                },
                "fixtureSha256": {
                    name: sha256_file(path) for name, path in sorted(fixtures.items())
                },
                "visual": await visual_evidence(
                    fixtures,
                    cast(CacheStore, store),
                    cast(CacheStore, UnavailableCacheStore()),
                    redis,
                    key_prefix,
                    args,
                ),
                "audio": await audio_evidence(
                    fixtures,
                    cast(CacheStore, store),
                    cast(CacheStore, UnavailableCacheStore()),
                    redis,
                    key_prefix,
                    args,
                ),
            }
        validate_report(report)
        encoded = json.dumps(report, indent=2, sort_keys=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
        print(encoded)
    finally:
        keys = [key async for key in redis.scan_iter(f"{key_prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


async def visual_evidence(
    fixtures: dict[str, Path],
    store: CacheStore,
    outage_store: CacheStore,
    redis: Any,
    key_prefix: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    real = ClipVisualEncoder(
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.visual_batch_size,
    )
    disabled_counter = CountingVisualEncoder(real)
    disabled = cached_visual(disabled_counter, store, False, key_prefix, args)
    disabled_first_started = perf_counter()
    disabled_first = await disabled.encode_frames([frame(fixtures["imageOriginal"])])
    disabled_first_ms = elapsed_ms(disabled_first_started)
    disabled_second_started = perf_counter()
    disabled_second = await disabled.encode_frames([frame(fixtures["imageOriginal"])])
    disabled_second_ms = elapsed_ms(disabled_second_started)

    counter = CountingVisualEncoder(real)
    cached = cached_visual(counter, store, True, key_prefix, args)
    before = metric_snapshot()
    commands_before = redis_command_snapshot()
    cold_started = perf_counter()
    cold = await cached.encode_frames(
        [
            frame(fixtures["imageOriginal"]),
            frame(fixtures["imageDuplicate"]),
            frame(fixtures["imageOriginal"]),
        ]
    )
    cold_ms = elapsed_ms(cold_started)
    after_cold = metric_snapshot()
    commands_after_cold = redis_command_snapshot()
    cold_inputs = counter.inputs
    warm_started = perf_counter()
    warm = await cached.encode_frames([frame(fixtures["imageDuplicate"])])
    warm_ms = elapsed_ms(warm_started)
    after_warm = metric_snapshot()
    commands_after_warm = redis_command_snapshot()
    warm_inputs = counter.inputs - cold_inputs

    shutil.copyfile(fixtures["imageChanged"], fixtures["imageDuplicate"])
    changed_started = perf_counter()
    changed = await cached.encode_frames([frame(fixtures["imageDuplicate"])])
    changed_ms = elapsed_ms(changed_started)
    after_change = metric_snapshot()
    commands_after_change = redis_command_snapshot()
    changed_inputs = counter.inputs - cold_inputs - warm_inputs

    original_hash = sha256_file(fixtures["imageOriginal"])
    original_key = cached.key_for_content_hash(original_hash)
    await redis.set(original_key, b"corrupt", ex=60)
    inputs_before_corrupt = counter.inputs
    corrupt_started = perf_counter()
    repaired = await cached.encode_frames([frame(fixtures["imageOriginal"])])
    corrupt_ms = elapsed_ms(corrupt_started)
    after_corrupt = metric_snapshot()
    commands_after_corrupt = redis_command_snapshot()
    repaired_raw = await redis.get(original_key)

    outage_counter = CountingVisualEncoder(real)
    outage = cached_visual(outage_counter, outage_store, True, f"{key_prefix}:outage", args)
    outage_started = perf_counter()
    outage_result = await outage.encode_frames([frame(fixtures["imageOriginal"])])
    outage_ms = elapsed_ms(outage_started)
    return {
        "disabledEncoderInputs": disabled_counter.inputs,
        "disabledEquivalent": vectors_equal(disabled_first, disabled_second),
        "disabledDurationsMs": [disabled_first_ms, disabled_second_ms],
        "coldEncoderInputs": cold_inputs,
        "coldDurationMs": cold_ms,
        "duplicateEquivalent": cold[0] == cold[1] == cold[2],
        "warmEncoderInputs": warm_inputs,
        "warmDurationMs": warm_ms,
        "warmEquivalent": vectors_equal(warm, [cold[0]]),
        "byteChangeEncoderInputs": changed_inputs,
        "byteChangeDurationMs": changed_ms,
        "byteChangeDifferent": not vectors_equal(changed, [cold[0]]),
        "corruptEncoderInputs": counter.inputs - inputs_before_corrupt,
        "corruptDurationMs": corrupt_ms,
        "corruptEquivalent": vectors_equal(repaired, [cold[0]]),
        "repairedMagic": magic(repaired_raw),
        "redisOutageEncoderInputs": outage_counter.inputs,
        "redisOutageDurationMs": outage_ms,
        "redisOutageEquivalent": vectors_equal(outage_result, disabled_first),
        "vectorSha256": {
            "original": sha256_json(cold[0]),
            "changed": sha256_json(changed[0]),
        },
        "metricDelta": {
            "cold": metric_delta(before, after_cold),
            "warm": metric_delta(after_cold, after_warm),
            "byteChange": metric_delta(after_warm, after_change),
            "corrupt": metric_delta(after_change, after_corrupt),
        },
        "redisCommandDelta": {
            "cold": redis_command_delta(commands_before, commands_after_cold),
            "warm": redis_command_delta(commands_after_cold, commands_after_warm),
            "byteChange": redis_command_delta(commands_after_warm, commands_after_change),
            "corrupt": redis_command_delta(commands_after_change, commands_after_corrupt),
        },
        "redis": await redis_namespace_stats(
            redis,
            f"{key_prefix}:ai:visual-embedding:*",
        ),
    }


async def audio_evidence(
    fixtures: dict[str, Path],
    store: CacheStore,
    outage_store: CacheStore,
    redis: Any,
    key_prefix: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    real = MsClapAudioEncoder(
        device=args.device,
        embedding_dim=1024,
        batch_size=args.audio_batch_size,
    )
    disabled_counter = CountingAudioEncoder(real)
    disabled = cached_audio(disabled_counter, store, False, key_prefix)
    disabled_first_started = perf_counter()
    disabled_first = await disabled.encode_audio_clips([clip(fixtures["audioOriginal"])])
    disabled_first_ms = elapsed_ms(disabled_first_started)
    disabled_second_started = perf_counter()
    disabled_second = await disabled.encode_audio_clips([clip(fixtures["audioOriginal"])])
    disabled_second_ms = elapsed_ms(disabled_second_started)

    counter = CountingAudioEncoder(real)
    cached = cached_audio(counter, store, True, key_prefix)
    before = metric_snapshot()
    commands_before = redis_command_snapshot()
    cold_started = perf_counter()
    cold = await cached.encode_audio_clips(
        [
            clip(fixtures["audioOriginal"]),
            clip(fixtures["audioDuplicate"]),
            clip(fixtures["audioOriginal"]),
        ]
    )
    cold_ms = elapsed_ms(cold_started)
    after_cold = metric_snapshot()
    commands_after_cold = redis_command_snapshot()
    cold_inputs = counter.inputs
    warm_started = perf_counter()
    warm = await cached.encode_audio_clips([clip(fixtures["audioDuplicate"])])
    warm_ms = elapsed_ms(warm_started)
    after_warm = metric_snapshot()
    commands_after_warm = redis_command_snapshot()
    warm_inputs = counter.inputs - cold_inputs

    shutil.copyfile(fixtures["audioChanged"], fixtures["audioDuplicate"])
    changed_started = perf_counter()
    changed = await cached.encode_audio_clips([clip(fixtures["audioDuplicate"])])
    changed_ms = elapsed_ms(changed_started)
    after_change = metric_snapshot()
    commands_after_change = redis_command_snapshot()
    changed_inputs = counter.inputs - cold_inputs - warm_inputs

    original_hash = sha256_file(fixtures["audioOriginal"])
    original_key = cached.key_for_content_hash(original_hash)
    await redis.set(original_key, b"corrupt", ex=60)
    inputs_before_corrupt = counter.inputs
    corrupt_started = perf_counter()
    repaired = await cached.encode_audio_clips([clip(fixtures["audioOriginal"])])
    corrupt_ms = elapsed_ms(corrupt_started)
    after_corrupt = metric_snapshot()
    commands_after_corrupt = redis_command_snapshot()
    repaired_raw = await redis.get(original_key)

    outage_counter = CountingAudioEncoder(real)
    outage = cached_audio(outage_counter, outage_store, True, f"{key_prefix}:outage")
    outage_started = perf_counter()
    outage_result = await outage.encode_audio_clips([clip(fixtures["audioOriginal"])])
    outage_ms = elapsed_ms(outage_started)
    return {
        "disabledEncoderInputs": disabled_counter.inputs,
        "disabledEquivalent": vectors_equal(disabled_first, disabled_second),
        "disabledDurationsMs": [disabled_first_ms, disabled_second_ms],
        "coldEncoderInputs": cold_inputs,
        "coldDurationMs": cold_ms,
        "duplicateEquivalent": cold[0] == cold[1] == cold[2],
        "warmEncoderInputs": warm_inputs,
        "warmDurationMs": warm_ms,
        "warmEquivalent": vectors_equal(warm, [cold[0]]),
        "byteChangeEncoderInputs": changed_inputs,
        "byteChangeDurationMs": changed_ms,
        "byteChangeDifferent": not vectors_equal(changed, [cold[0]]),
        "corruptEncoderInputs": counter.inputs - inputs_before_corrupt,
        "corruptDurationMs": corrupt_ms,
        "corruptEquivalent": vectors_equal(repaired, [cold[0]]),
        "repairedMagic": magic(repaired_raw),
        "redisOutageEncoderInputs": outage_counter.inputs,
        "redisOutageDurationMs": outage_ms,
        "redisOutageEquivalent": vectors_equal(outage_result, disabled_first),
        "vectorSha256": {
            "original": sha256_json(cold[0]),
            "changed": sha256_json(changed[0]),
        },
        "metricDelta": {
            "cold": metric_delta(before, after_cold),
            "warm": metric_delta(after_cold, after_warm),
            "byteChange": metric_delta(after_warm, after_change),
            "corrupt": metric_delta(after_change, after_corrupt),
        },
        "redisCommandDelta": {
            "cold": redis_command_delta(commands_before, commands_after_cold),
            "warm": redis_command_delta(commands_after_cold, commands_after_warm),
            "byteChange": redis_command_delta(commands_after_warm, commands_after_change),
            "corrupt": redis_command_delta(commands_after_change, commands_after_corrupt),
        },
        "redis": await redis_namespace_stats(
            redis,
            f"{key_prefix}:ai:audio-embedding:*",
        ),
    }


def cached_visual(
    counter: CountingVisualEncoder,
    cache: CacheStore,
    enabled: bool,
    key_prefix: str,
    args: argparse.Namespace,
) -> CachedVisualEmbeddingEncoder:
    return CachedVisualEmbeddingEncoder(
        cast(VisualEncoder, counter),
        cache=cache,
        enabled=enabled,
        model_id=visual_model_id(args.clip_model_name, args.clip_pretrained, 512),
        dimension=512,
        key_prefix=key_prefix,
    )


def cached_audio(
    counter: CountingAudioEncoder,
    cache: CacheStore,
    enabled: bool,
    key_prefix: str,
) -> CachedAudioEmbeddingEncoder:
    return CachedAudioEmbeddingEncoder(
        cast(AudioEmbeddingEncoder, counter),
        cache=cache,
        enabled=enabled,
        model_id=audio_model_id(1024),
        dimension=1024,
        key_prefix=key_prefix,
    )


def create_fixtures(directory: Path) -> dict[str, Path]:
    image_original = directory / "original.png"
    image_duplicate = directory / "duplicate.png"
    image_changed = directory / "changed.png"
    Image.new("RGB", (64, 64), (24, 96, 192)).save(image_original, format="PNG")
    shutil.copyfile(image_original, image_duplicate)
    Image.new("RGB", (64, 64), (192, 64, 24)).save(image_changed, format="PNG")

    audio_original = directory / "original.wav"
    audio_duplicate = directory / "duplicate.wav"
    audio_changed = directory / "changed.wav"
    write_pcm_wav(audio_original, frequency=440.0)
    shutil.copyfile(audio_original, audio_duplicate)
    write_pcm_wav(audio_changed, frequency=660.0)
    return {
        "imageOriginal": image_original,
        "imageDuplicate": image_duplicate,
        "imageChanged": image_changed,
        "audioOriginal": audio_original,
        "audioDuplicate": audio_duplicate,
        "audioChanged": audio_changed,
    }


def write_pcm_wav(path: Path, *, frequency: float) -> None:
    sample_rate = 48_000
    duration_seconds = 1.0
    sample_count = round(sample_rate * duration_seconds)
    frames = bytearray()
    for index in range(sample_count):
        value = round(12_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(bytes(frames))


def frame(path: Path) -> SampledFrame:
    return SampledFrame(shot=evidence_shot(), timestamp_seconds=0.5, path=path)


def clip(path: Path) -> ShotAudioClip:
    return ShotAudioClip(
        shot=evidence_shot(),
        path=path,
        clip_start_seconds=0.0,
        clip_end_seconds=1.0,
        clip_duration_seconds=1.0,
    )


def evidence_shot() -> DetectedShot:
    return DetectedShot(
        shot_id="evidence:shot:000000",
        media_id="evidence",
        shot_index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        duration_seconds=1.0,
    )


def metric_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for name in _COUNTERS:
        for outcome in _OUTCOMES:
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


async def redis_namespace_stats(redis: Any, pattern: str) -> dict[str, Any]:
    keys = [key async for key in redis.scan_iter(pattern)]
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


def validate_report(report: dict[str, Any]) -> None:
    for media, expected_magic in (("visual", "KVEV"), ("audio", "KAEV")):
        evidence = report[media]
        checks = (
            evidence["disabledEncoderInputs"] == 2,
            evidence["disabledEquivalent"],
            evidence["coldEncoderInputs"] == 1,
            evidence["duplicateEquivalent"],
            evidence["warmEncoderInputs"] == 0,
            evidence["warmEquivalent"],
            evidence["byteChangeEncoderInputs"] == 1,
            evidence["byteChangeDifferent"],
            evidence["corruptEncoderInputs"] == 1,
            evidence["corruptEquivalent"],
            evidence["repairedMagic"] == expected_magic,
            evidence["redisOutageEncoderInputs"] == 1,
            evidence["redisOutageEquivalent"],
            evidence["redisCommandDelta"]["cold"]
            == {"mget:success": 1.0, "pipeline_set:success": 1.0},
            evidence["redisCommandDelta"]["warm"] == {"mget:success": 1.0},
            evidence["redis"]["keyCardinality"] == 2,
            evidence["redis"]["payloadBytes"]["total"] > 0,
            evidence["redis"]["memoryBytes"] > 0,
        )
        if not all(checks):
            raise RuntimeError(
                f"{media} embedding cache evidence validation failed: "
                f"{json.dumps(evidence, sort_keys=True)}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def vectors_equal(left: list[list[float]], right: list[list[float]]) -> bool:
    return len(left) == len(right) and all(
        len(left_vector) == len(right_vector)
        and all(
            math.isclose(float(left_value), float(right_value), rel_tol=1e-6, abs_tol=1e-7)
            for left_value, right_value in zip(left_vector, right_vector, strict=True)
        )
        for left_vector, right_vector in zip(left, right, strict=True)
    )


def magic(value: Any) -> str | None:
    return bytes(value[:4]).decode("ascii") if isinstance(value, bytes) else None


def redact_redis_url(url: str) -> str:
    if "@" not in url:
        return url
    return f"{url.split('://', 1)[0]}://{url.rsplit('@', 1)[1]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/cache/phase3-visual-audio-local.json"),
    )
    parser.add_argument("--redis-url", default="redis://localhost:6379/15")
    parser.add_argument("--redis-timeout-seconds", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--visual-batch-size", type=int, default=16)
    parser.add_argument("--audio-batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
