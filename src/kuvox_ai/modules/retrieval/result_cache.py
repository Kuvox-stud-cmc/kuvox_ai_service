"""Short-lived cache decorator for trusted video-editor retrieval results."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from kuvox_ai.cache import CacheKeyFactory, CacheStore, JsonCacheCodec, ReadOutcome, WriteOutcome
from kuvox_ai.distributed_lock import LockStore
from kuvox_ai.metrics import RETRIEVAL_CACHE_OPERATIONS, RETRIEVAL_CACHE_PAYLOAD_BYTES
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.modules.retrieval.models import (
    RetrievalQuery,
    VideoEditorRetrievalQuery,
    VideoEditorRetrievalResult,
)
from kuvox_ai.modules.retrieval.service import RetrievalService
from kuvox_ai.schemas import RetrievalResult
from kuvox_ai.single_flight import Probe, ProbeOutcome, SingleFlight
from kuvox_ai.text_embedding_cache import canonicalize_text_embedding_input

_SCOPE_REVISION = re.compile(r"^[0-9a-f]{64}$")
_MODALITY_ORDER = ("visual", "transcript", "audio", "ocr")
_SCHEMA_ID = "video-editor-retrieval-v1"


class CachedVideoEditorRetrievalService:
    def __init__(
        self,
        wrapped: RetrievalService,
        *,
        cache: CacheStore,
        locks: LockStore,
        enabled: bool,
        single_flight_enabled: bool,
        ttl_seconds: int = 60,
        max_payload_bytes: int = 1_048_576,
        key_prefix: str = "kuvox:v1",
        transcript_collection_name: str = "shots_transcript",
        ocr_collection_name: str = "shots_ocr",
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_embedding_dimension: int = 384,
        lock_ttl_seconds: float = 30,
        lock_wait_seconds: float = 15,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._wrapped = wrapped
        self._cache = cache
        self._enabled = enabled
        self._single_flight_enabled = enabled and single_flight_enabled
        self._ttl_seconds = ttl_seconds
        self._max_payload_bytes = max_payload_bytes
        self._keys = CacheKeyFactory(key_prefix)
        self._codec = JsonCacheCodec(schema_version=1)
        config = {
            "schema": _SCHEMA_ID,
            "collections": {
                "transcript": transcript_collection_name,
                "ocr": ocr_collection_name,
            },
            "searchable_modalities": ["transcript", "ocr"],
            "embedding": {
                "model": text_embedding_model_name,
                "dimension": text_embedding_dimension,
                "normalization": "nfc-lf-v1",
            },
            "rrf_constant": 60,
            "candidate_behavior": "per-modality-limit-equals-top-k-v1",
            "evidence_limit": 4,
            "graph_expansion": "per-result-previous-next-v1",
            "reranker": "none-v1",
        }
        canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
        self.configuration_identity = self._keys.sha256(canonical_config)
        self._single_flight = SingleFlight(
            locks,
            component="retrieval",
            lock_ttl_seconds=lock_ttl_seconds,
            wait_seconds=lock_wait_seconds,
            poll_seconds=lock_poll_seconds,
        )

    @property
    def _text_encoder(self) -> TextEmbeddingEncoder:
        """Compatibility/introspection surface for existing wiring checks."""
        return self._wrapped._text_encoder

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return await self._wrapped.retrieve(query)

    async def retrieve_video_editor(
        self,
        query: VideoEditorRetrievalQuery,
    ) -> VideoEditorRetrievalResult:
        cache_key = self.key_for(query)
        if not self._enabled or cache_key is None:
            RETRIEVAL_CACHE_OPERATIONS.labels("bypass").inc()
            return await self._wrapped.retrieve_video_editor(query)

        initial = await self._read(cache_key, query)
        if initial.outcome is ProbeOutcome.HIT and initial.value is not None:
            return initial.value

        async def authoritative() -> VideoEditorRetrievalResult:
            authoritative_result = await self._wrapped.retrieve_video_editor_authoritative(query)
            if (
                authoritative_result.complete
                and authoritative_result.result.results
                and not authoritative_result.result.warnings
            ):
                await self._write(cache_key, authoritative_result.result)
            return authoritative_result.result

        if initial.outcome is ProbeOutcome.FAILURE:
            return await authoritative()
        if not self._single_flight_enabled:
            return await authoritative()
        return await self._single_flight.run(
            cache_key,
            probe=lambda: self._read(cache_key, query),
            authoritative=authoritative,
        )

    def key_for(self, query: VideoEditorRetrievalQuery) -> str | None:
        scope_revision = query.scope_revision
        if not isinstance(scope_revision, str) or not _SCOPE_REVISION.fullmatch(scope_revision):
            return None
        project_id = query.project_id.strip().lower()
        media_ids = sorted({item.strip().lower() for item in query.media_ids if item.strip()})
        canonical_query = canonicalize_text_embedding_input(query.query)
        modalities = [name for name in _MODALITY_ORDER if name in set(query.modalities)]
        if not project_id or not media_ids or not canonical_query.strip() or not modalities:
            return None
        request_identity = json.dumps(
            {
                "project": project_id,
                "media": media_ids,
                "query": canonical_query,
                "modalities": modalities,
                "top_k": query.top_k,
                "expand_graph": query.expand_graph,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return self._keys.create(
            "ai",
            "retrieval",
            "schema",
            _SCHEMA_ID,
            "config",
            self.configuration_identity,
            "scope",
            scope_revision,
            self._keys.sha256(request_identity),
        )

    async def _read(
        self,
        key: str,
        query: VideoEditorRetrievalQuery,
    ) -> Probe[VideoEditorRetrievalResult]:
        try:
            read = await self._cache.get(key)
        except Exception:  # noqa: BLE001 - optional cache fails open
            RETRIEVAL_CACHE_OPERATIONS.labels("error").inc()
            return Probe(ProbeOutcome.FAILURE)
        if read.outcome is ReadOutcome.ERROR or read.outcome is ReadOutcome.BYPASS:
            RETRIEVAL_CACHE_OPERATIONS.labels("error").inc()
            return Probe(ProbeOutcome.FAILURE)
        if read.outcome is ReadOutcome.MISS or read.value is None:
            RETRIEVAL_CACHE_OPERATIONS.labels("miss").inc()
            return Probe(ProbeOutcome.MISS)
        if len(read.value) > self._max_payload_bytes:
            RETRIEVAL_CACHE_OPERATIONS.labels("oversized").inc()
            return Probe(ProbeOutcome.MISS)
        RETRIEVAL_CACHE_PAYLOAD_BYTES.labels("read").observe(len(read.value))
        payload = self._codec.decode(read.value)
        if (
            not isinstance(payload, dict)
            or payload.get("configuration_identity") != self.configuration_identity
        ):
            await self._discard_corrupt(key)
            return Probe(ProbeOutcome.MISS)
        try:
            result = VideoEditorRetrievalResult.model_validate(payload.get("result"))
        except ValidationError:
            await self._discard_corrupt(key)
            return Probe(ProbeOutcome.MISS)
        trusted_media = {item.strip().lower() for item in query.media_ids if item.strip()}
        if (
            not result.results
            or result.warnings
            or result.project_id.strip().lower() != query.project_id.strip().lower()
            or canonicalize_text_embedding_input(result.query)
            != canonicalize_text_embedding_input(query.query)
            or len(result.results) > query.top_k
            or any(item.media_id.strip().lower() not in trusted_media for item in result.results)
        ):
            await self._discard_corrupt(key)
            return Probe(ProbeOutcome.MISS)
        RETRIEVAL_CACHE_OPERATIONS.labels("hit").inc()
        return Probe(ProbeOutcome.HIT, result)

    async def _discard_corrupt(self, key: str) -> None:
        RETRIEVAL_CACHE_OPERATIONS.labels("corrupt").inc()
        try:
            await self._cache.delete(key)
        except Exception:  # noqa: BLE001
            RETRIEVAL_CACHE_OPERATIONS.labels("error").inc()

    async def _write(self, key: str, result: VideoEditorRetrievalResult) -> None:
        payload = self._codec.encode(
            {
                "configuration_identity": self.configuration_identity,
                "result": result.model_dump(mode="json", by_alias=True),
            }
        )
        RETRIEVAL_CACHE_PAYLOAD_BYTES.labels("write").observe(len(payload))
        if len(payload) > self._max_payload_bytes:
            RETRIEVAL_CACHE_OPERATIONS.labels("oversized").inc()
            return
        try:
            outcome = await self._cache.set(key, payload, self._ttl_seconds)
        except Exception:  # noqa: BLE001
            RETRIEVAL_CACHE_OPERATIONS.labels("error").inc()
            return
        if outcome is WriteOutcome.SUCCESS:
            RETRIEVAL_CACHE_OPERATIONS.labels("write").inc()
        elif outcome is WriteOutcome.ERROR:
            RETRIEVAL_CACHE_OPERATIONS.labels("error").inc()
        else:
            RETRIEVAL_CACHE_OPERATIONS.labels("bypass").inc()
