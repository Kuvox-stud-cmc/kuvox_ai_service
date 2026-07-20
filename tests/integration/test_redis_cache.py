from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from kuvox_ai.cache import (
    CacheWrite,
    CircuitBreaker,
    JsonCacheCodec,
    ReadOutcome,
    RedisCacheStore,
    TtlJitter,
    WriteOutcome,
)
from kuvox_ai.distributed_lock import LockAcquireOutcome, LockHandle, RedisLockStore
from kuvox_ai.infrastructure.redis_client import RedisClient
from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.audio_embedding_cache import CachedAudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.audio_encoder import AudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.text_embedding_cache import CachedIngestionTextEmbeddingEncoder
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.modules.ingestion.visual_embedding_cache import CachedVisualEmbeddingEncoder
from kuvox_ai.modules.ingestion.visual_encoder import VisualEncoder
from kuvox_ai.modules.retrieval.models import (
    VideoEditorRetrievalQuery,
    VideoEditorRetrievalResult,
    VideoEditorShotResult,
)
from kuvox_ai.modules.retrieval.query_embedding_cache import (
    CachedQueryTextEmbeddingEncoder,
    QueryEmbeddingBinaryCodec,
)
from kuvox_ai.modules.retrieval.result_cache import CachedVideoEditorRetrievalService
from kuvox_ai.modules.retrieval.service import AuthoritativeVideoEditorRetrieval, RetrievalService

REDIS_TEST_URL = os.getenv("KUVOX_TEST_REDIS_URL", "redis://localhost:6379/15")
REDIS_API_ACL_TEST_URL = os.getenv("KUVOX_TEST_REDIS_API_ACL_URL")
REDIS_AI_ACL_TEST_URL = os.getenv("KUVOX_TEST_REDIS_AI_ACL_URL")


class RecordingEncoder:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return [[0.25, -0.5, 0.75] for _ in texts]


class RecordingVisualEncoder:
    def __init__(self) -> None:
        self.inputs: list[list[Path]] = []

    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
        self.inputs.append([frame.path for frame in frames])
        return [[0.25, -0.5, 0.75] for _ in frames]


class RecordingAudioEncoder:
    def __init__(self) -> None:
        self.inputs: list[list[Path]] = []

    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        self.inputs.append([clip.path for clip in clips])
        return [[0.5, -0.5] for _ in clips]


def integration_shot() -> DetectedShot:
    return DetectedShot(
        shot_id="integration:shot:000000",
        media_id="integration",
        shot_index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        duration_seconds=1.0,
    )


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def utcnow(self) -> datetime:
        return datetime(2026, 7, 17, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.now


class SwitchableRedis:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.fail = True

    async def get(self, key: str) -> bytes | None:
        if self.fail:
            raise ConnectionError("simulated Redis outage")
        value = await self.client.get(key)
        return None if value is None else bytes(value)

    async def set(self, key: str, value: bytes, *, ex: int) -> bool:
        if self.fail:
            raise ConnectionError("simulated Redis outage")
        return bool(await self.client.set(key, value, ex=ex))

    async def delete(self, key: str) -> int:
        if self.fail:
            raise ConnectionError("simulated Redis outage")
        return int(await self.client.delete(key))


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_real_redis_api_ai_acl_namespaces_are_isolated() -> None:
    if REDIS_API_ACL_TEST_URL is None or REDIS_AI_ACL_TEST_URL is None:
        pytest.skip("ACL Redis URLs are not configured")
    api = aioredis.from_url(REDIS_API_ACL_TEST_URL, decode_responses=False)
    ai = aioredis.from_url(REDIS_AI_ACL_TEST_URL, decode_responses=False)
    try:
        api_key = "kuvox:v1:api:acl-test"
        ai_key = "kuvox:v1:ai:acl-test"
        release = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('del', KEYS[1])
        end
        return 0
        """
        assert await api.set(api_key, b"api-owner", nx=True, px=5_000) is True
        assert await ai.set(ai_key, b"ai-owner", nx=True, px=5_000) is True
        assert 0 < await api.pttl(api_key) <= 5_000
        assert 0 < await ai.pttl(ai_key) <= 5_000
        assert await api.eval(release, 1, api_key, b"not-owner") == 0
        assert await ai.eval(release, 1, ai_key, b"not-owner") == 0
        with pytest.raises(ResponseError):
            await api.set("kuvox:v1:ai:forbidden", b"no")
        with pytest.raises(ResponseError):
            await ai.set("kuvox:v1:api:forbidden", b"no")
        assert await api.eval(release, 1, api_key, b"api-owner") == 1
        assert await ai.eval(release, 1, ai_key, b"ai-owner") == 1
        with pytest.raises(ResponseError):
            await api.eval("return redis.call('del', KEYS[1])", 1, "kuvox:v1:ai:forbidden")
        with pytest.raises(ResponseError):
            await api.flushdb()
        with pytest.raises(ResponseError):
            await ai.flushdb()
    finally:
        await api.delete("kuvox:v1:api:acl-test")
        await ai.delete("kuvox:v1:ai:acl-test")
        await api.aclose()
        await ai.aclose()


class SlowRetrieval:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve_video_editor_authoritative(
        self, query: VideoEditorRetrievalQuery
    ) -> AuthoritativeVideoEditorRetrieval:
        self.calls += 1
        await asyncio.sleep(0.05)
        return AuthoritativeVideoEditorRetrieval(
            VideoEditorRetrievalResult(
                project_id=query.project_id,
                query=query.query,
                results=[
                    VideoEditorShotResult(
                        shot_id="shot-1",
                        media_id=query.media_ids[0],
                        start_seconds=0,
                        end_seconds=1,
                        score=1,
                    )
                ],
            ),
            True,
        )


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_real_redis_set_nx_px_lua_and_two_instance_retrieval_single_flight() -> None:
    first_redis = RedisClient(REDIS_TEST_URL)
    second_redis = RedisClient(REDIS_TEST_URL)
    await first_redis.connect()
    await second_redis.connect()
    try:
        if not await first_redis.health_check():
            pytest.skip("Redis is unavailable")
        await first_redis.client.flushdb()
        first_locks = RedisLockStore(first_redis, operation_timeout_seconds=1)
        second_locks = RedisLockStore(second_redis, operation_timeout_seconds=1)
        acquired = await first_locks.acquire("retrieval", "ttl-contract", 30)
        assert acquired.outcome is LockAcquireOutcome.ACQUIRED
        assert acquired.handle is not None
        assert 29_000 <= await first_redis.client.pttl(acquired.handle.key) <= 30_000
        assert (
            await second_locks.release(LockHandle(acquired.handle.key, b"not-the-owner")) is False
        )
        assert await first_locks.release(acquired.handle) is True

        source = SlowRetrieval()
        first_service = CachedVideoEditorRetrievalService(
            cast(RetrievalService, source),
            cache=RedisCacheStore(first_redis, jitter=TtlJitter(0)),
            locks=first_locks,
            enabled=True,
            single_flight_enabled=True,
            lock_poll_seconds=0.005,
        )
        second_service = CachedVideoEditorRetrievalService(
            cast(RetrievalService, source),
            cache=RedisCacheStore(second_redis, jitter=TtlJitter(0)),
            locks=second_locks,
            enabled=True,
            single_flight_enabled=True,
            lock_poll_seconds=0.005,
        )
        query = VideoEditorRetrievalQuery(
            projectId="project-1",
            mediaIds=["media-1"],
            query="identical cold request",
            modalities=["transcript"],
            expandGraph=False,
            scopeRevision="a" * 64,
        )
        results = await asyncio.gather(
            *(
                (first_service if index % 2 == 0 else second_service).retrieve_video_editor(query)
                for index in range(16)
            )
        )
        assert source.calls == 1
        assert all(result == results[0] for result in results)

        encoder = RecordingEncoder()
        first_query_encoder = CachedQueryTextEmbeddingEncoder(
            cast(TextEmbeddingEncoder, encoder),
            cache=RedisCacheStore(first_redis, jitter=TtlJitter(0)),
            enabled=True,
            model_id="model/single-flight-integration",
            dimension=3,
            lock_store=first_locks,
            single_flight_enabled=True,
            lock_poll_seconds=0.005,
        )
        second_query_encoder = CachedQueryTextEmbeddingEncoder(
            cast(TextEmbeddingEncoder, encoder),
            cache=RedisCacheStore(second_redis, jitter=TtlJitter(0)),
            enabled=True,
            model_id="model/single-flight-integration",
            dimension=3,
            lock_store=second_locks,
            single_flight_enabled=True,
            lock_poll_seconds=0.005,
        )
        vectors = await asyncio.gather(
            *(
                (first_query_encoder if index % 2 == 0 else second_query_encoder).encode_texts(
                    ["identical encoder input"]
                )
                for index in range(16)
            )
        )
        assert encoder.inputs == [["identical encoder input"]]
        assert all(vector == vectors[0] for vector in vectors)
    finally:
        if await first_redis.health_check():
            await first_redis.client.flushdb()
        await second_redis.close()
        await first_redis.close()


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_real_redis_raw_bytes_ttl_delete_and_connection_reuse() -> None:
    client = aioredis.from_url(
        REDIS_TEST_URL,
        decode_responses=False,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )
    available = False
    try:
        try:
            await client.ping()
            available = True
        except Exception as exc:  # noqa: BLE001 - unavailable integration service skips
            pytest.skip(f"Redis is unavailable: {type(exc).__name__}")
        store = RedisCacheStore(client, jitter=TtlJitter(0))
        await client.flushdb()
        bulk_outcomes = await store.set_many(
            [
                CacheWrite("kuvox:bulk:a", b"first", 10),
                CacheWrite("kuvox:bulk:b", b"second", 20),
            ]
        )
        assert bulk_outcomes == [WriteOutcome.SUCCESS, WriteOutcome.SUCCESS]
        bulk_reads = await store.get_many(["kuvox:bulk:b", "kuvox:bulk:missing", "kuvox:bulk:a"])
        assert [read.outcome for read in bulk_reads] == [
            ReadOutcome.HIT,
            ReadOutcome.MISS,
            ReadOutcome.HIT,
        ]
        assert [read.value for read in bulk_reads] == [b"second", None, b"first"]
        assert await client.ttl("kuvox:bulk:a") == 10
        assert await client.ttl("kuvox:bulk:b") == 20
        assert await store.set("kuvox:test", b"raw\x00bytes", 1) is WriteOutcome.SUCCESS
        assert (await store.get("kuvox:test")).value == b"raw\x00bytes"
        await asyncio.sleep(1.1)
        assert (await store.get("kuvox:test")).outcome is ReadOutcome.MISS
        assert await store.set("kuvox:test", b"again", 30) is WriteOutcome.SUCCESS
        codec = JsonCacheCodec()
        assert await store.set("kuvox:json", codec.encode({"ok": True}), 30) is WriteOutcome.SUCCESS
        json_value = await store.get("kuvox:json")
        assert codec.decode(json_value.value or b"") == {"ok": True}
        assert await store.delete("kuvox:test") is WriteOutcome.SUCCESS
        assert (await store.get("kuvox:test")).outcome is ReadOutcome.MISS
        await client.flushdb()
        assert [
            read.outcome for read in await store.get_many(["kuvox:bulk:a", "kuvox:bulk:b"])
        ] == [
            ReadOutcome.MISS,
            ReadOutcome.MISS,
        ]
        assert await store.set_many([CacheWrite("kuvox:bulk:a", b"recovered", 30)]) == [
            WriteOutcome.SUCCESS
        ]
    finally:
        if available:
            await client.delete("kuvox:test", "kuvox:json", "kuvox:bulk:a", "kuvox:bulk:b")
        await client.aclose()


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_process_redis_wrapper_supports_bulk_cache_commands() -> None:
    redis = RedisClient(
        REDIS_TEST_URL,
        connect_timeout_seconds=0.5,
        operation_timeout_seconds=0.5,
    )
    await redis.connect()
    try:
        if not await redis.health_check():
            pytest.skip("Redis is unavailable")
        store = RedisCacheStore(redis, jitter=TtlJitter(0))
        assert await store.set_many(
            [
                CacheWrite("kuvox:wrapper:a", b"a", 30),
                CacheWrite("kuvox:wrapper:b", b"b", 30),
            ]
        ) == [WriteOutcome.SUCCESS, WriteOutcome.SUCCESS]
        reads = await store.get_many(["kuvox:wrapper:b", "kuvox:wrapper:a"])
        assert [read.value for read in reads] == [b"b", b"a"]
    finally:
        if await redis.health_check():
            await redis.delete("kuvox:wrapper:a")
            await redis.delete("kuvox:wrapper:b")
        await redis.close()


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_real_redis_shared_embedding_binary_expiry_corrupt_repair_and_legacy_promotion() -> (
    None
):
    client = aioredis.from_url(
        REDIS_TEST_URL,
        decode_responses=False,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )
    available = False
    try:
        try:
            await client.ping()
            available = True
        except Exception as exc:  # noqa: BLE001 - unavailable integration service skips
            pytest.skip(f"Redis is unavailable: {type(exc).__name__}")
        await client.flushdb()
        store = RedisCacheStore(client, jitter=TtlJitter(0))
        encoder = RecordingEncoder()
        cached = CachedQueryTextEmbeddingEncoder(
            cast(TextEmbeddingEncoder, encoder),
            cache=store,
            enabled=True,
            model_id="model/integration",
            dimension=3,
            ttl_seconds=1,
        )

        first = await cached.encode_texts(["real Redis query"])
        second = await cached.encode_texts(["real Redis query"])
        assert second == first
        assert encoder.inputs == [["real Redis query"]]
        key = cached.key_for_text("real Redis query")
        raw = await client.get(key)
        assert isinstance(raw, bytes)
        assert raw.startswith(b"KTEV\x01")

        await client.set(key, b"corrupt", ex=30)
        repaired = await cached.encode_texts(["real Redis query"])
        assert repaired == first
        assert encoder.inputs == [["real Redis query"], ["real Redis query"]]
        repaired_raw = await client.get(key)
        assert isinstance(repaired_raw, bytes)
        assert repaired_raw.startswith(b"KTEV\x01")

        await asyncio.sleep(1.1)
        assert await client.get(key) is None

        legacy_payload = QueryEmbeddingBinaryCodec(
            model_id="model/integration", dimension=3
        ).encode([0.5, 0.25, -0.75])
        assert legacy_payload is not None
        await client.set(cached.legacy_key_for_text("legacy Redis query"), legacy_payload, ex=30)
        promoted = await cached.encode_texts(["legacy Redis query"])
        assert promoted == [[0.5, 0.25, -0.75]]
        assert encoder.inputs == [["real Redis query"], ["real Redis query"]]
        promoted_raw = await client.get(cached.key_for_text("legacy Redis query"))
        assert isinstance(promoted_raw, bytes)
        assert promoted_raw.startswith(b"KTEV\x01")

        ingestion_encoder = RecordingEncoder()
        ingestion = CachedIngestionTextEmbeddingEncoder(
            cast(TextEmbeddingEncoder, ingestion_encoder),
            cache=store,
            enabled=True,
            model_id="model/integration",
            dimension=3,
            ttl_seconds=604_800,
            legacy_read_enabled=False,
        )
        await ingestion.encode_texts(["seven day shared entry"])
        ttl = await client.ttl(ingestion.key_for_text("seven day shared entry"))
        assert 604_799 <= ttl <= 604_800

        query_reuse = await cached.encode_texts(["seven day shared entry"])
        assert query_reuse == [[0.25, -0.5, 0.75]]
        assert encoder.inputs == [["real Redis query"], ["real Redis query"]]
    finally:
        if available:
            await client.flushdb()
        await client.aclose()


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_real_redis_visual_audio_round_trip_expiry_repair_and_independent_keys(
    tmp_path: Path,
) -> None:
    client = aioredis.from_url(
        REDIS_TEST_URL,
        decode_responses=False,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )
    available = False
    try:
        try:
            await client.ping()
            available = True
        except Exception as exc:  # noqa: BLE001 - unavailable integration service skips
            pytest.skip(f"Redis is unavailable: {type(exc).__name__}")
        await client.flushdb()
        store = RedisCacheStore(client, jitter=TtlJitter(0))
        image_path = tmp_path / "fixture.png"
        audio_path = tmp_path / "fixture.wav"
        image_path.write_bytes(b"deterministic png bytes")
        audio_path.write_bytes(b"deterministic pcm wav bytes")
        frame = SampledFrame(
            shot=integration_shot(),
            timestamp_seconds=0.5,
            path=image_path,
        )
        clip = ShotAudioClip(
            shot=integration_shot(),
            path=audio_path,
            clip_start_seconds=0.0,
            clip_end_seconds=1.0,
            clip_duration_seconds=1.0,
        )
        visual_authoritative = RecordingVisualEncoder()
        audio_authoritative = RecordingAudioEncoder()
        visual = CachedVisualEmbeddingEncoder(
            cast(VisualEncoder, visual_authoritative),
            cache=store,
            enabled=True,
            model_id="openclip:integration",
            dimension=3,
            ttl_seconds=1,
        )
        audio = CachedAudioEmbeddingEncoder(
            cast(AudioEmbeddingEncoder, audio_authoritative),
            cache=store,
            enabled=True,
            model_id="msclap:integration",
            dimension=2,
            ttl_seconds=30,
        )

        assert await visual.encode_frames([frame, frame]) == [
            [0.25, -0.5, 0.75],
            [0.25, -0.5, 0.75],
        ]
        assert await visual.encode_frames([frame]) == [[0.25, -0.5, 0.75]]
        assert visual_authoritative.inputs == [[image_path]]
        visual_keys = [key async for key in client.scan_iter("kuvox:v1:ai:visual-embedding:*")]
        assert len(visual_keys) == 1
        visual_key = visual_keys[0]
        visual_raw = await client.get(visual_key)
        assert isinstance(visual_raw, bytes)
        assert visual_raw.startswith(b"KVEV\x01")

        await client.set(visual_key, b"corrupt", ex=30)
        assert await visual.encode_frames([frame]) == [[0.25, -0.5, 0.75]]
        assert visual_authoritative.inputs == [[image_path], [image_path]]
        repaired = await client.get(visual_key)
        assert isinstance(repaired, bytes)
        assert repaired.startswith(b"KVEV\x01")

        assert await audio.encode_audio_clips([clip]) == [[0.5, -0.5]]
        assert await audio.encode_audio_clips([clip]) == [[0.5, -0.5]]
        assert audio_authoritative.inputs == [[audio_path]]
        audio_keys = [key async for key in client.scan_iter("kuvox:v1:ai:audio-embedding:*")]
        assert len(audio_keys) == 1
        audio_key = audio_keys[0]
        audio_raw = await client.get(audio_key)
        assert isinstance(audio_raw, bytes)
        assert audio_raw.startswith(b"KAEV\x01")
        assert visual_key != audio_key

        await asyncio.sleep(1.1)
        assert await client.get(visual_key) is None
        assert await client.get(audio_key) is not None
    finally:
        if available:
            await client.flushdb()
        await client.aclose()


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_real_redis_store_recovers_after_circuit_half_open_attempt() -> None:
    client = aioredis.from_url(
        REDIS_TEST_URL,
        decode_responses=False,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )
    available = False
    try:
        try:
            await client.ping()
            available = True
        except Exception as exc:  # noqa: BLE001 - unavailable integration service skips
            pytest.skip(f"Redis is unavailable: {type(exc).__name__}")
        await client.flushdb()
        clock = ManualClock()
        circuit = CircuitBreaker(clock=clock, failure_threshold=1, open_seconds=10)
        redis = SwitchableRedis(client)
        store = RedisCacheStore(redis, circuit=circuit, jitter=TtlJitter(0))

        assert (await store.get("kuvox:half-open")).outcome is ReadOutcome.ERROR
        assert circuit.state == "open"
        assert (await store.get("kuvox:half-open")).outcome is ReadOutcome.BYPASS
        redis.fail = False
        clock.now = 10
        assert (await store.get("kuvox:half-open")).outcome is ReadOutcome.MISS
        assert circuit.state == "closed"
        assert await store.set("kuvox:half-open", b"recovered", 30) is WriteOutcome.SUCCESS
        assert (await store.get("kuvox:half-open")).value == b"recovered"
    finally:
        if available:
            await client.delete("kuvox:half-open")
        await client.aclose()
