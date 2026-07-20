"""Shared text-embedding cache configured for video-editor queries."""

from __future__ import annotations

from kuvox_ai.cache import CacheStore
from kuvox_ai.distributed_lock import LockStore
from kuvox_ai.metrics import (
    QUERY_EMBEDDING_CACHE_DURATION,
    QUERY_EMBEDDING_CACHE_OPERATIONS,
    QUERY_EMBEDDING_CACHE_PAYLOAD_BYTES,
    QUERY_EMBEDDING_ENCODER_INPUTS,
)
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.text_embedding_cache import (
    LEGACY_QUERY_EMBEDDING_MAGIC,
    TEXT_EMBEDDING_NORMALIZATION_ID,
    CachedTextEmbeddingEncoder,
    TextEmbeddingBinaryCodec,
    TextEmbeddingCacheMetrics,
    TextEmbeddingCacheOutcome,
    TextEmbeddingDecode,
    canonicalize_text_embedding_input,
)

QUERY_EMBEDDING_NORMALIZATION_ID = TEXT_EMBEDDING_NORMALIZATION_ID
QueryEmbeddingCacheOutcome = TextEmbeddingCacheOutcome
QueryEmbeddingDecode = TextEmbeddingDecode
canonicalize_query_text = canonicalize_text_embedding_input


class QueryEmbeddingBinaryCodec(TextEmbeddingBinaryCodec):
    """Legacy Phase 1 KQEV codec retained for migration and compatibility."""

    def __init__(self, *, model_id: str, dimension: int) -> None:
        super().__init__(
            magic=LEGACY_QUERY_EMBEDDING_MAGIC,
            model_id=model_id,
            dimension=dimension,
        )


class CachedQueryTextEmbeddingEncoder(CachedTextEmbeddingEncoder):
    """Shared cache decorator configured with query-specific metrics."""

    def __init__(
        self,
        wrapped: TextEmbeddingEncoder,
        *,
        cache: CacheStore,
        enabled: bool,
        model_id: str,
        dimension: int,
        ttl_seconds: int = 604_800,
        key_prefix: str = "kuvox:v1",
        legacy_read_enabled: bool = True,
        lock_store: LockStore | None = None,
        single_flight_enabled: bool = False,
        lock_ttl_seconds: float = 30,
        lock_wait_seconds: float = 15,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            wrapped,
            cache=cache,
            metrics=TextEmbeddingCacheMetrics(
                operations=QUERY_EMBEDDING_CACHE_OPERATIONS,
                duration=QUERY_EMBEDDING_CACHE_DURATION,
                payload_bytes=QUERY_EMBEDDING_CACHE_PAYLOAD_BYTES,
                encoder_inputs=QUERY_EMBEDDING_ENCODER_INPUTS,
            ),
            enabled=enabled,
            model_id=model_id,
            dimension=dimension,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
            legacy_read_enabled=legacy_read_enabled,
            lock_store=lock_store,
            single_flight_enabled=single_flight_enabled,
            lock_ttl_seconds=lock_ttl_seconds,
            lock_wait_seconds=lock_wait_seconds,
            lock_poll_seconds=lock_poll_seconds,
        )
