"""Public interface of the ingestion module."""

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

__all__ = [
    "AudioMetadata",
    "DetectedShot",
    "ImageMetadata",
    "IngestionCompleted",
    "IngestionFailed",
    "IngestionRequested",
    "IngestionService",
    "OptimizedObject",
    "VideoMetadata",
]
