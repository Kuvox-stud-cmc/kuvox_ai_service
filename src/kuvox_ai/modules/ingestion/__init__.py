"""Public interface of the ingestion module."""

from kuvox_ai.modules.ingestion.models import (
    DetectedShot,
    IngestionCompleted,
    IngestionFailed,
    IngestionRequested,
    OptimizedObject,
    VideoMetadata,
)
from kuvox_ai.modules.ingestion.service import IngestionService

__all__ = [
    "DetectedShot",
    "IngestionCompleted",
    "IngestionFailed",
    "IngestionRequested",
    "IngestionService",
    "OptimizedObject",
    "VideoMetadata",
]
