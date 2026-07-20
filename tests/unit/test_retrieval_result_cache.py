from __future__ import annotations

import asyncio
from typing import cast

import pytest

from kuvox_ai.cache import CacheRead, CacheStore, ReadOutcome, WriteOutcome
from kuvox_ai.distributed_lock import (
    LockAcquire,
    LockAcquireOutcome,
    LockHandle,
    LockStore,
)
from kuvox_ai.modules.retrieval.models import (
    VideoEditorRetrievalQuery,
    VideoEditorRetrievalResult,
    VideoEditorShotResult,
)
from kuvox_ai.modules.retrieval.result_cache import CachedVideoEditorRetrievalService
from kuvox_ai.modules.retrieval.service import AuthoritativeVideoEditorRetrieval, RetrievalService


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.gets = 0
        self.sets = 0

    async def get(self, key: str) -> CacheRead:
        self.gets += 1
        value = self.values.get(key)
        return (
            CacheRead(ReadOutcome.HIT, value) if value is not None else CacheRead(ReadOutcome.MISS)
        )

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        assert ttl_seconds == 60
        self.sets += 1
        self.values[key] = value
        return WriteOutcome.SUCCESS

    async def delete(self, key: str) -> WriteOutcome:
        self.values.pop(key, None)
        return WriteOutcome.SUCCESS


class MemoryLocks:
    def __init__(self) -> None:
        self.owners: dict[str, bytes] = {}

    async def acquire(self, component: str, cache_key: str, ttl_seconds: float) -> LockAcquire:
        del ttl_seconds
        key = f"lock:{component}:{cache_key}"
        owner = str(id(asyncio.current_task())).encode()
        if key in self.owners:
            return LockAcquire(LockAcquireOutcome.CONTENDED, LockHandle(key, owner))
        self.owners[key] = owner
        return LockAcquire(LockAcquireOutcome.ACQUIRED, LockHandle(key, owner))

    async def is_locked(self, handle_key: str) -> bool | None:
        return handle_key in self.owners

    async def release(self, handle: LockHandle) -> bool:
        if self.owners.get(handle.key) != handle.owner:
            return False
        self.owners.pop(handle.key, None)
        return True


class FakeAuthoritative:
    def __init__(
        self,
        *,
        complete: bool = True,
        delay: float = 0,
        empty: bool = False,
        warnings: list[str] | None = None,
    ) -> None:
        self.calls = 0
        self.complete = complete
        self.delay = delay
        self.empty = empty
        self.warnings = warnings or []

    async def retrieve_video_editor_authoritative(
        self, query: VideoEditorRetrievalQuery
    ) -> AuthoritativeVideoEditorRetrieval:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        result = VideoEditorRetrievalResult(
            project_id=query.project_id,
            query=query.query,
            results=[]
            if self.empty
            else [
                VideoEditorShotResult(
                    shot_id="shot-1",
                    media_id=query.media_ids[0],
                    start_seconds=0,
                    end_seconds=1,
                    score=1,
                )
            ],
            warnings=self.warnings,
        )
        return AuthoritativeVideoEditorRetrieval(result, self.complete)

    async def retrieve_video_editor(
        self, query: VideoEditorRetrievalQuery
    ) -> VideoEditorRetrievalResult:
        return (await self.retrieve_video_editor_authoritative(query)).result


def query(**changes: object) -> VideoEditorRetrievalQuery:
    values: dict[str, object] = {
        "projectId": "project-1",
        "mediaIds": ["media-1"],
        "query": "Cafe\u0301\r\nquery",
        "modalities": ["ocr", "transcript", "ocr"],
        "topK": 8,
        "expandGraph": True,
        "scopeRevision": "a" * 64,
    }
    values.update(changes)
    return VideoEditorRetrievalQuery.model_validate(values)


def cached(
    authoritative: FakeAuthoritative,
    cache: MemoryCache,
    *,
    single_flight: bool = False,
) -> CachedVideoEditorRetrievalService:
    return CachedVideoEditorRetrievalService(
        cast(RetrievalService, authoritative),
        cache=cast(CacheStore, cache),
        locks=cast(LockStore, MemoryLocks()),
        enabled=True,
        single_flight_enabled=single_flight,
        lock_wait_seconds=1,
        lock_poll_seconds=0.001,
    )


def configured_key(*, collection: str, model: str) -> str | None:
    service = CachedVideoEditorRetrievalService(
        cast(RetrievalService, FakeAuthoritative()),
        cache=cast(CacheStore, MemoryCache()),
        locks=cast(LockStore, MemoryLocks()),
        enabled=True,
        single_flight_enabled=False,
        transcript_collection_name=collection,
        text_embedding_model_name=model,
    )
    return service.key_for(query())


def test_conflicting_scope_revision_aliases_bypass_cache_identity() -> None:
    request = VideoEditorRetrievalQuery.model_validate(
        {
            "projectId": "project-1",
            "mediaIds": ["media-1"],
            "query": "query",
            "scopeRevision": "a" * 64,
            "scope_revision": "b" * 64,
        }
    )
    assert request.scope_revision is None


@pytest.mark.asyncio
async def test_cold_then_warm_cache_and_canonical_key_isolation() -> None:
    store = MemoryCache()
    source = FakeAuthoritative()
    service = cached(source, store)
    request = query()

    assert await service.retrieve_video_editor(request) == await service.retrieve_video_editor(
        request
    )
    assert source.calls == 1
    assert service.key_for(request) == service.key_for(
        query(query="Café\nquery", modalities=["transcript", "ocr"])
    )
    assert service.key_for(request) != service.key_for(query(projectId="project-2"))
    assert service.key_for(request) != service.key_for(query(scopeRevision="b" * 64))
    assert service.key_for(request) != service.key_for(query(topK=9))
    assert service.key_for(request) != service.key_for(query(expandGraph=False))
    assert configured_key(collection="transcript-v1", model="model-v1") != configured_key(
        collection="transcript-v2", model="model-v1"
    )
    assert configured_key(collection="transcript-v1", model="model-v1") != configured_key(
        collection="transcript-v1", model="model-v2"
    )


@pytest.mark.asyncio
async def test_invalid_scope_and_incomplete_results_bypass_writes() -> None:
    store = MemoryCache()
    source = FakeAuthoritative(complete=False)
    service = cached(source, store)

    await service.retrieve_video_editor(query(scopeRevision="bad"))
    await service.retrieve_video_editor(query(scopeRevision="bad"))
    await service.retrieve_video_editor(query())
    await service.retrieve_video_editor(query())

    assert source.calls == 4
    assert store.sets == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        FakeAuthoritative(empty=True),
        FakeAuthoritative(warnings=["Qdrant unavailable"]),
        FakeAuthoritative(complete=False),
    ],
)
async def test_empty_warning_and_partial_authoritative_results_are_not_cached(
    source: FakeAuthoritative,
) -> None:
    store = MemoryCache()
    service = cached(source, store)

    await service.retrieve_video_editor(query())
    await service.retrieve_video_editor(query())

    assert source.calls == 2
    assert store.sets == 0


@pytest.mark.asyncio
async def test_oversized_authoritative_result_is_returned_but_not_cached() -> None:
    store = MemoryCache()
    source = FakeAuthoritative()
    service = CachedVideoEditorRetrievalService(
        cast(RetrievalService, source),
        cache=cast(CacheStore, store),
        locks=cast(LockStore, MemoryLocks()),
        enabled=True,
        single_flight_enabled=False,
        max_payload_bytes=1,
    )

    first = await service.retrieve_video_editor(query())
    second = await service.retrieve_video_editor(query())

    assert first == second
    assert source.calls == 2
    assert store.sets == 0


@pytest.mark.asyncio
async def test_corrupt_payload_is_recomputed_and_repaired() -> None:
    store = MemoryCache()
    source = FakeAuthoritative()
    service = cached(source, store)
    key = service.key_for(query())
    assert key is not None
    store.values[key] = b"not-json"

    result = await service.retrieve_video_editor(query())

    assert result.results
    assert source.calls == 1
    assert store.values[key] != b"not-json"


@pytest.mark.asyncio
async def test_sixteen_concurrent_cold_requests_execute_one_pipeline() -> None:
    store = MemoryCache()
    source = FakeAuthoritative(delay=0.02)
    service = cached(source, store, single_flight=True)

    results = await asyncio.gather(*(service.retrieve_video_editor(query()) for _ in range(16)))

    assert source.calls == 1
    assert all(result == results[0] for result in results)
