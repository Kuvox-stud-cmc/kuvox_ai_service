"""Public interface of the ingestion module."""

from kuvox_ai.modules.ingestion.audio_embedding_cache import CachedAudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.models import (
    AudioMetadata,
    DetectedShot,
    ImageMetadata,
    IngestionCompleted,
    IngestionFailed,
    IngestionRequested,
    OptimizedObject,
    VideoMetadata,
)
from kuvox_ai.modules.ingestion.service import IngestionService
from kuvox_ai.modules.ingestion.text_embedding_cache import CachedIngestionTextEmbeddingEncoder
from kuvox_ai.modules.ingestion.visual_embedding_cache import CachedVisualEmbeddingEncoder

__all__ = [
    "AudioMetadata",
    "CachedAudioEmbeddingEncoder",
    "CachedIngestionTextEmbeddingEncoder",
    "CachedVisualEmbeddingEncoder",
    "DetectedShot",
    "ImageMetadata",
    "IngestionCompleted",
    "IngestionFailed",
    "IngestionRequested",
    "IngestionService",
    "OptimizedObject",
    "VideoMetadata",
]
