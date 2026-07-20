"""Content-addressed cache for OpenCLIP visual embeddings."""

from __future__ import annotations

from kuvox_ai.cache import CacheStore
from kuvox_ai.config import Settings
from kuvox_ai.file_embedding_cache import (
    CachedFileEmbeddingEncoder,
    FileEmbeddingCacheMetrics,
    installed_versions,
)
from kuvox_ai.metrics import (
    VISUAL_EMBEDDING_CACHE_DURATION,
    VISUAL_EMBEDDING_CACHE_OPERATIONS,
    VISUAL_EMBEDDING_CACHE_PAYLOAD_BYTES,
    VISUAL_EMBEDDING_ENCODER_INPUTS,
)
from kuvox_ai.modules.ingestion.frame_sampler import FRAME_SAMPLING_CONTRACT_ID, SampledFrame
from kuvox_ai.modules.ingestion.visual_encoder import ClipVisualEncoder, VisualEncoder

VISUAL_EMBEDDING_MAGIC = b"KVEV"
VISUAL_IMAGE_PREPROCESSING_ID = "pil-open-rgb-openclip-transform-l2-clamp-1e-12-v1"
VISUAL_STANDALONE_IMAGE_INPUT_CONTRACT_ID = "standalone-image-exact-file-bytes-v1"
VISUAL_EMBEDDING_PIPELINE_ID = (
    f"preprocess={VISUAL_IMAGE_PREPROCESSING_ID};"
    f"frame-sampling={FRAME_SAMPLING_CONTRACT_ID};"
    f"standalone-input={VISUAL_STANDALONE_IMAGE_INPUT_CONTRACT_ID}"
)


class CachedVisualEmbeddingEncoder:
    def __init__(
        self,
        wrapped: VisualEncoder,
        *,
        cache: CacheStore,
        enabled: bool,
        model_id: str,
        dimension: int,
        pipeline_id: str = VISUAL_EMBEDDING_PIPELINE_ID,
        ttl_seconds: int = 86_400,
        key_prefix: str = "kuvox:v1",
    ) -> None:
        self._cached = CachedFileEmbeddingEncoder(
            wrapped.encode_frames,
            path_of=lambda frame: frame.path,
            cache=cache,
            metrics=FileEmbeddingCacheMetrics(
                operations=VISUAL_EMBEDDING_CACHE_OPERATIONS,
                duration=VISUAL_EMBEDDING_CACHE_DURATION,
                payload_bytes=VISUAL_EMBEDDING_CACHE_PAYLOAD_BYTES,
                encoder_inputs=VISUAL_EMBEDDING_ENCODER_INPUTS,
            ),
            enabled=enabled,
            namespace="visual-embedding",
            magic=VISUAL_EMBEDDING_MAGIC,
            model_id=model_id,
            pipeline_id=pipeline_id,
            dimension=dimension,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
        )

    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
        return await self._cached.encode_items(frames)

    def key_for_content_hash(self, content_hash: str) -> str:
        return self._cached.key_for_content_hash(content_hash)


def build_visual_embedding_encoder(
    settings: Settings,
    cache: CacheStore,
) -> CachedVisualEmbeddingEncoder:
    return CachedVisualEmbeddingEncoder(
        ClipVisualEncoder(
            model_name=settings.clip_model_name,
            pretrained=settings.clip_pretrained,
            device=settings.clip_device,
            batch_size=settings.clip_batch_size,
        ),
        cache=cache,
        enabled=settings.cache_enabled and settings.visual_embedding_cache_enabled,
        model_id=visual_model_id(
            settings.clip_model_name,
            settings.clip_pretrained,
            settings.visual_embedding_dim,
        ),
        dimension=settings.visual_embedding_dim,
        ttl_seconds=settings.visual_embedding_cache_ttl_seconds,
        key_prefix=settings.cache_key_prefix,
    )


def visual_model_id(model_name: str, pretrained: str, dimension: int) -> str:
    versions = installed_versions("open-clip-torch", "torch", "torchvision", "Pillow")
    return (
        f"openclip:model={model_name};pretrained={pretrained};dimension={dimension};"
        f"versions={versions}"
    )
