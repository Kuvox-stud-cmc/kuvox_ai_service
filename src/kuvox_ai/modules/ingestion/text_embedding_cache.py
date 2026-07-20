"""Shared text-embedding cache configured for ingestion transcript and OCR text."""

from __future__ import annotations

from kuvox_ai.cache import CacheStore
from kuvox_ai.config import Settings
from kuvox_ai.metrics import (
    INGESTION_TEXT_EMBEDDING_CACHE_DURATION,
    INGESTION_TEXT_EMBEDDING_CACHE_OPERATIONS,
    INGESTION_TEXT_EMBEDDING_CACHE_PAYLOAD_BYTES,
    INGESTION_TEXT_EMBEDDING_ENCODER_INPUTS,
)
from kuvox_ai.modules.ingestion.text_encoder import (
    SentenceTransformerTextEncoder,
    TextEmbeddingEncoder,
)
from kuvox_ai.text_embedding_cache import CachedTextEmbeddingEncoder, TextEmbeddingCacheMetrics


class CachedIngestionTextEmbeddingEncoder(CachedTextEmbeddingEncoder):
    """Shared cache decorator configured with ingestion-specific metrics."""

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
    ) -> None:
        super().__init__(
            wrapped,
            cache=cache,
            metrics=TextEmbeddingCacheMetrics(
                operations=INGESTION_TEXT_EMBEDDING_CACHE_OPERATIONS,
                duration=INGESTION_TEXT_EMBEDDING_CACHE_DURATION,
                payload_bytes=INGESTION_TEXT_EMBEDDING_CACHE_PAYLOAD_BYTES,
                encoder_inputs=INGESTION_TEXT_EMBEDDING_ENCODER_INPUTS,
            ),
            enabled=enabled,
            model_id=model_id,
            dimension=dimension,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
            legacy_read_enabled=legacy_read_enabled,
        )


def build_ingestion_text_embedding_encoder(
    settings: Settings,
    cache: CacheStore,
) -> CachedIngestionTextEmbeddingEncoder:
    return CachedIngestionTextEmbeddingEncoder(
        SentenceTransformerTextEncoder(
            model_name=settings.text_embedding_model_name,
            device=settings.text_embedding_device,
            batch_size=settings.text_embedding_batch_size,
        ),
        cache=cache,
        enabled=settings.cache_enabled and settings.ingestion_text_embedding_cache_enabled,
        model_id=settings.text_embedding_model_name,
        dimension=settings.text_embedding_dim,
        ttl_seconds=settings.text_embedding_cache_ttl_seconds,
        key_prefix=settings.cache_key_prefix,
        legacy_read_enabled=settings.text_embedding_cache_legacy_read_enabled,
    )
