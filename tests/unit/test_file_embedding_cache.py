from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from prometheus_client import generate_latest

import kuvox_ai.file_embedding_cache as file_cache_module
from kuvox_ai.cache import CacheRead, CacheStore, CacheWrite, ReadOutcome, WriteOutcome
from kuvox_ai.file_embedding_cache import FileFingerprint, fingerprint_file
from kuvox_ai.modules.ingestion.audio import (
    AUDIO_CLIP_BOUNDARY_CONTRACT_ID,
    FULL_AUDIO_EXTRACTION_CONTRACT_ID,
    SHOT_AUDIO_EXTRACTION_CONTRACT_ID,
    ShotAudioClip,
)
from kuvox_ai.modules.ingestion.audio_embedding_cache import (
    AUDIO_EMBEDDING_PIPELINE_ID,
    CachedAudioEmbeddingEncoder,
    audio_model_id,
)
from kuvox_ai.modules.ingestion.audio_encoder import AudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.frame_sampler import FRAME_SAMPLING_CONTRACT_ID, SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot
from kuvox_ai.modules.ingestion.visual_embedding_cache import (
    VISUAL_EMBEDDING_PIPELINE_ID,
    VISUAL_STANDALONE_IMAGE_INPUT_CONTRACT_ID,
    CachedVisualEmbeddingEncoder,
    visual_model_id,
)
from kuvox_ai.modules.ingestion.visual_encoder import VisualEncoder


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, bytes, int]] = []
        self.fail_get = False
        self.fail_set = False

    async def get(self, key: str) -> CacheRead:
        self.gets.append(key)
        if self.fail_get:
            raise ConnectionError("cache unavailable")
        value = self.values.get(key)
        return CacheRead(ReadOutcome.HIT, value) if value else CacheRead(ReadOutcome.MISS)

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        if self.fail_set:
            raise ConnectionError("cache unavailable")
        self.sets.append((key, value, ttl_seconds))
        self.values[key] = value
        return WriteOutcome.SUCCESS

    async def delete(self, key: str) -> WriteOutcome:
        self.values.pop(key, None)
        return WriteOutcome.SUCCESS


class BulkMemoryCache(MemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_gets: list[list[str]] = []
        self.bulk_sets: list[list[CacheWrite]] = []

    async def get_many(self, keys: list[str]) -> list[CacheRead]:
        self.bulk_gets.append(list(keys))
        if self.fail_get:
            raise ConnectionError("cache unavailable")
        return [
            CacheRead(ReadOutcome.HIT, self.values[key])
            if key in self.values
            else CacheRead(ReadOutcome.MISS)
            for key in keys
        ]

    async def set_many(self, entries: list[CacheWrite]) -> list[WriteOutcome]:
        if self.fail_set:
            raise ConnectionError("cache unavailable")
        self.bulk_sets.append(list(entries))
        for entry in entries:
            self.values[entry.key] = entry.value
        return [WriteOutcome.SUCCESS for _ in entries]


class ByteVisualEncoder:
    def __init__(self, *, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[Path]] = []
        self.mutate_during_encode = False
        self.error: BaseException | None = None
        self.vectors: list[list[float]] | None = None

    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
        self.calls.append([frame.path for frame in frames])
        if self.error is not None:
            raise self.error
        values: list[list[float]] = []
        for frame in frames:
            data = frame.path.read_bytes()
            base = float(sum(data) % 97)
            values.append([base + index for index in range(self.dimension)])
            if self.mutate_during_encode:
                frame.path.write_bytes(data + b"changed")
        return self.vectors if self.vectors is not None else values


class ByteAudioEncoder:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        self.calls.append([clip.path for clip in clips])
        return [[0.25, -0.75] for _ in clips]


def shot() -> DetectedShot:
    return DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        duration_seconds=1.0,
    )


def frame(path: Path) -> SampledFrame:
    return SampledFrame(shot=shot(), timestamp_seconds=0.5, path=path)


def clip(path: Path) -> ShotAudioClip:
    return ShotAudioClip(
        shot=shot(),
        path=path,
        clip_start_seconds=0.0,
        clip_end_seconds=1.0,
        clip_duration_seconds=1.0,
    )


def cached_visual(
    wrapped: ByteVisualEncoder,
    cache: MemoryCache,
    *,
    enabled: bool = True,
    model_id: str = "openclip:test",
    pipeline_id: str = "pipeline-v1",
    dimension: int = 3,
    ttl_seconds: int = 86_400,
) -> CachedVisualEmbeddingEncoder:
    return CachedVisualEmbeddingEncoder(
        cast(VisualEncoder, wrapped),
        cache=cast(CacheStore, cache),
        enabled=enabled,
        model_id=model_id,
        pipeline_id=pipeline_id,
        dimension=dimension,
        ttl_seconds=ttl_seconds,
    )


async def test_empty_and_disabled_batches_bypass_hashing_and_cache(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"pixels")
    authoritative = ByteVisualEncoder()
    store = MemoryCache()
    cached = cached_visual(authoritative, store, enabled=False)

    assert await cached.encode_frames([]) == []
    assert await cached.encode_frames([frame(path)]) == [[79.0, 80.0, 81.0]]
    assert authoritative.calls == [[path]]
    assert store.gets == []
    assert store.sets == []


async def test_exact_bytes_share_entries_and_duplicates_are_encoded_once(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"identical bytes")
    second.write_bytes(b"identical bytes")
    authoritative = ByteVisualEncoder()
    store = MemoryCache()
    cached = cached_visual(authoritative, store)

    cold = await cached.encode_frames([frame(first), frame(second), frame(first)])
    warm = await cached.encode_frames([frame(second), frame(first)])

    assert cold[0] == cold[1] == cold[2]
    assert warm == cold[:2]
    assert authoritative.calls == [[first]]
    assert len(store.sets) == 1
    assert store.sets[0][1].startswith(b"KVEV\x01")
    assert store.sets[0][2] == 86_400


async def test_byte_change_causes_miss_and_mixed_batches_preserve_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    authoritative = ByteVisualEncoder()
    store = MemoryCache()
    cached = cached_visual(authoritative, store)

    first_vector = (await cached.encode_frames([frame(first)]))[0]
    second.write_bytes(b"second changed")
    mixed = await cached.encode_frames([frame(second), frame(first), frame(second)])

    assert mixed[1] == pytest.approx(first_vector)
    assert mixed[0] == mixed[2]
    assert authoritative.calls == [[first], [second]]


async def test_unique_file_content_uses_one_bulk_read_and_one_bulk_write(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    third = tmp_path / "third.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    third.write_bytes(b"third")
    authoritative = ByteVisualEncoder()
    store = BulkMemoryCache()
    cached = cached_visual(authoritative, store)
    first_vector = (await cached.encode_frames([frame(first)]))[0]
    authoritative.calls.clear()
    store.bulk_gets.clear()
    store.bulk_sets.clear()

    result = await cached.encode_frames([frame(second), frame(first), frame(third), frame(second)])

    assert result[1] == pytest.approx(first_vector)
    assert result[0] == result[3]
    assert authoritative.calls == [[second, third]]
    assert len(store.bulk_gets) == 1
    assert len(store.bulk_gets[0]) == 3
    assert len(store.bulk_sets) == 1
    assert len(store.bulk_sets[0]) == 2
    assert store.gets == []
    assert store.sets == []


def test_model_pipeline_and_dimension_isolate_keys() -> None:
    store = MemoryCache()
    digest = hashlib.sha256(b"same").hexdigest()
    first = cached_visual(ByteVisualEncoder(), store, model_id="model-a")
    model = cached_visual(ByteVisualEncoder(), store, model_id="model-b")
    pipeline = cached_visual(ByteVisualEncoder(), store, pipeline_id="pipeline-v2")
    dimension = cached_visual(ByteVisualEncoder(dimension=4), store, dimension=4)

    keys = {
        first.key_for_content_hash(digest),
        model.key_for_content_hash(digest),
        pipeline.key_for_content_hash(digest),
        dimension.key_for_content_hash(digest),
    }
    assert len(keys) == 4
    for key in keys:
        assert ":identity:" in key
        assert key.endswith(f":content:{digest}")


def test_production_visual_audio_identities_bind_all_preprocessing_contracts() -> None:
    assert FRAME_SAMPLING_CONTRACT_ID in VISUAL_EMBEDDING_PIPELINE_ID
    assert VISUAL_STANDALONE_IMAGE_INPUT_CONTRACT_ID in VISUAL_EMBEDDING_PIPELINE_ID
    assert FULL_AUDIO_EXTRACTION_CONTRACT_ID in AUDIO_EMBEDDING_PIPELINE_ID
    assert SHOT_AUDIO_EXTRACTION_CONTRACT_ID in AUDIO_EMBEDDING_PIPELINE_ID
    assert AUDIO_CLIP_BOUNDARY_CONTRACT_ID in AUDIO_EMBEDDING_PIPELINE_ID

    assert visual_model_id("model-a", "weights-a", 512) != visual_model_id(
        "model-b", "weights-a", 512
    )
    assert visual_model_id("model-a", "weights-a", 512) != visual_model_id(
        "model-a", "weights-b", 512
    )
    assert visual_model_id("model-a", "weights-a", 512) != visual_model_id(
        "model-a", "weights-a", 768
    )
    assert audio_model_id(1024) != audio_model_id(512)


async def test_corrupt_values_and_cache_outages_recompute_and_repair(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    authoritative = ByteVisualEncoder()
    store = MemoryCache()
    cached = cached_visual(authoritative, store)
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    key = cached.key_for_content_hash(content_hash)
    store.values[key] = b"corrupt"

    repaired = await cached.encode_frames([frame(path)])
    assert repaired
    assert store.values[key].startswith(b"KVEV\x01")

    store.values.clear()
    store.fail_get = True
    store.fail_set = True
    fallback = await cached.encode_frames([frame(path)])
    assert fallback == repaired
    assert len(authoritative.calls) == 2


@pytest.mark.parametrize("vector", [[1.0, 2.0], [1.0, float("nan"), 3.0]])
async def test_invalid_authoritative_vectors_are_returned_but_not_cached(
    tmp_path: Path,
    vector: list[float],
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    authoritative = ByteVisualEncoder()
    authoritative.vectors = [vector]
    store = MemoryCache()
    cached = cached_visual(authoritative, store)

    assert await cached.encode_frames([frame(path)]) == [vector]
    assert store.sets == []


async def test_missing_files_and_authoritative_exceptions_propagate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        await cached_visual(ByteVisualEncoder(), MemoryCache()).encode_frames([frame(missing)])

    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    authoritative = ByteVisualEncoder()
    authoritative.error = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        await cached_visual(authoritative, MemoryCache()).encode_frames([frame(path)])


async def test_cancellation_propagates_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    started = asyncio.Event()

    class BlockingEncoder(ByteVisualEncoder):
        async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
            started.set()
            await asyncio.Event().wait()
            return await super().encode_frames(frames)

    store = MemoryCache()
    task = asyncio.create_task(cached_visual(BlockingEncoder(), store).encode_frames([frame(path)]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.sets == []


async def test_hash_retries_once_after_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    original = cast(Callable[[Path], str], file_cache_module._sha256_file)
    calls = 0

    def mutate_once(target: Path) -> str:
        nonlocal calls
        calls += 1
        digest = original(target)
        if calls == 1:
            target.write_bytes(b"after and stable")
        return digest

    monkeypatch.setattr(file_cache_module, "_sha256_file", mutate_once)
    fingerprint = await fingerprint_file(path)

    assert fingerprint.stable is True
    assert calls == 2
    assert fingerprint.content_hash == hashlib.sha256(b"after and stable").hexdigest()


async def test_hash_concurrency_is_bounded_at_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"input-{index}.bin" for index in range(8)]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]))
    active = 0
    maximum = 0

    async def measured(path: Path) -> FileFingerprint:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        signature = await file_cache_module._stat_signature_async(path)
        data = await asyncio.to_thread(path.read_bytes)
        return FileFingerprint(hashlib.sha256(data).hexdigest(), signature, True)

    monkeypatch.setattr(file_cache_module, "fingerprint_file", measured)
    authoritative = ByteVisualEncoder()
    await cached_visual(authoritative, MemoryCache()).encode_frames([frame(path) for path in paths])

    assert maximum == 4


async def test_unstable_hash_and_encoding_mutation_never_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    signature = await file_cache_module._stat_signature_async(path)

    async def unstable(_: Path) -> FileFingerprint:
        return FileFingerprint("0" * 64, signature, False)

    monkeypatch.setattr(file_cache_module, "fingerprint_file", unstable)
    store = MemoryCache()
    authoritative = ByteVisualEncoder()
    assert await cached_visual(authoritative, store).encode_frames([frame(path)])
    assert store.sets == []

    monkeypatch.undo()
    store = MemoryCache()
    authoritative = ByteVisualEncoder()
    authoritative.mutate_during_encode = True
    assert await cached_visual(authoritative, store).encode_frames([frame(path)])
    assert store.sets == []


async def test_audio_cache_has_independent_key_and_kaev_payload(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"pcm bytes")
    authoritative = ByteAudioEncoder()
    store = MemoryCache()
    cached = CachedAudioEmbeddingEncoder(
        cast(AudioEmbeddingEncoder, authoritative),
        cache=cast(CacheStore, store),
        enabled=True,
        model_id="msclap:test",
        dimension=2,
    )

    assert await cached.encode_audio_clips([clip(path), clip(path)]) == [
        [0.25, -0.75],
        [0.25, -0.75],
    ]
    assert authoritative.calls == [[path]]
    assert store.sets[0][0].startswith("kuvox:v1:ai:audio-embedding:identity:")
    assert store.sets[0][1].startswith(b"KAEV\x01")


async def test_metrics_do_not_expose_paths_hashes_or_business_labels(tmp_path: Path) -> None:
    secret = tmp_path / "private-owner-project-media.png"
    secret.write_bytes(b"private")
    cached = cached_visual(ByteVisualEncoder(), MemoryCache())
    await cached.encode_frames([frame(secret)])
    await cached.encode_frames([frame(secret)])

    exposition = generate_latest().decode("utf-8")
    phase3_lines = "\n".join(
        line
        for line in exposition.splitlines()
        if "visual_embedding_cache" in line or "visual_embedding_encoder" in line
    )
    assert "kuvox_visual_embedding_cache_duration_seconds" in phase3_lines
    assert secret.name not in phase3_lines
    for forbidden in ("path=", "hash=", "media=", "model=", "owner=", "user="):
        assert forbidden not in phase3_lines
