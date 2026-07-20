"""Content-addressed cache for MSCLAP audio embeddings."""

from __future__ import annotations

from kuvox_ai.cache import CacheStore
from kuvox_ai.config import Settings
from kuvox_ai.file_embedding_cache import (
    CachedFileEmbeddingEncoder,
    FileEmbeddingCacheMetrics,
    installed_versions,
)
from kuvox_ai.metrics import (
    AUDIO_EMBEDDING_CACHE_DURATION,
    AUDIO_EMBEDDING_CACHE_OPERATIONS,
    AUDIO_EMBEDDING_CACHE_PAYLOAD_BYTES,
    AUDIO_EMBEDDING_ENCODER_INPUTS,
)
from kuvox_ai.modules.ingestion.audio import (
    AUDIO_CLIP_BOUNDARY_CONTRACT_ID,
    FULL_AUDIO_EXTRACTION_CONTRACT_ID,
    SHOT_AUDIO_EXTRACTION_CONTRACT_ID,
    ShotAudioClip,
)
from kuvox_ai.modules.ingestion.audio_encoder import AudioEmbeddingEncoder, MsClapAudioEncoder

AUDIO_EMBEDDING_MAGIC = b"KAEV"
AUDIO_PREPROCESSING_ID = (
    "soundfile-float32-always-2d-channel-mean-v1;"
    "librosa-model-native-resample-v1;"
    "repeat-then-truncate-model-duration-v1;l2-epsilon-1e-12-v1"
)
AUDIO_EMBEDDING_PIPELINE_ID = (
    f"preprocess={AUDIO_PREPROCESSING_ID};"
    f"full-audio={FULL_AUDIO_EXTRACTION_CONTRACT_ID};"
    f"shot-audio={SHOT_AUDIO_EXTRACTION_CONTRACT_ID};"
    f"clip-boundary={AUDIO_CLIP_BOUNDARY_CONTRACT_ID}"
)


class CachedAudioEmbeddingEncoder:
    def __init__(
        self,
        wrapped: AudioEmbeddingEncoder,
        *,
        cache: CacheStore,
        enabled: bool,
        model_id: str,
        dimension: int,
        pipeline_id: str = AUDIO_EMBEDDING_PIPELINE_ID,
        ttl_seconds: int = 86_400,
        key_prefix: str = "kuvox:v1",
    ) -> None:
        self._cached = CachedFileEmbeddingEncoder(
            wrapped.encode_audio_clips,
            path_of=lambda clip: clip.path,
            cache=cache,
            metrics=FileEmbeddingCacheMetrics(
                operations=AUDIO_EMBEDDING_CACHE_OPERATIONS,
                duration=AUDIO_EMBEDDING_CACHE_DURATION,
                payload_bytes=AUDIO_EMBEDDING_CACHE_PAYLOAD_BYTES,
                encoder_inputs=AUDIO_EMBEDDING_ENCODER_INPUTS,
            ),
            enabled=enabled,
            namespace="audio-embedding",
            magic=AUDIO_EMBEDDING_MAGIC,
            model_id=model_id,
            pipeline_id=pipeline_id,
            dimension=dimension,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
        )

    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        return await self._cached.encode_items(clips)

    def key_for_content_hash(self, content_hash: str) -> str:
        return self._cached.key_for_content_hash(content_hash)


def build_audio_embedding_encoder(
    settings: Settings,
    cache: CacheStore,
) -> CachedAudioEmbeddingEncoder:
    return CachedAudioEmbeddingEncoder(
        MsClapAudioEncoder(
            device=settings.audio_embedding_device,
            embedding_dim=settings.audio_embedding_dim,
            batch_size=settings.audio_embedding_batch_size,
        ),
        cache=cache,
        enabled=settings.cache_enabled and settings.audio_embedding_cache_enabled,
        model_id=audio_model_id(settings.audio_embedding_dim),
        dimension=settings.audio_embedding_dim,
        ttl_seconds=settings.audio_embedding_cache_ttl_seconds,
        key_prefix=settings.cache_key_prefix,
    )


def audio_model_id(dimension: int) -> str:
    versions = installed_versions("msclap", "torch", "librosa", "numpy", "soundfile")
    return (
        f"msclap:model=CLAP;config=package-default-audio-v1;dimension={dimension};"
        f"versions={versions}"
    )
