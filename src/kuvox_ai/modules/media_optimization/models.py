"""Message contracts for the media optimization pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MediaKind(StrEnum):
    video = "Video"
    audio = "Audio"
    image = "Image"


class MediaOptimizationRequested(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: str = Field(alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")

    media_id: str = Field(alias="mediaId")
    user_id: str = Field(alias="userId")

    bucket_name: str = Field(alias="bucketName")
    object_key: str = Field(alias="objectKey")

    content_type: str = Field(alias="contentType")
    original_file_name: str = Field(alias="originalFileName")
    size_bytes: int = Field(alias="sizeBytes")
    kind: MediaKind

    model_config = ConfigDict(populate_by_name=True)


class OptimizedObject(BaseModel):
    bucket_name: str = Field(alias="bucketName")
    object_key: str = Field(alias="objectKey")
    content_type: str = Field(alias="contentType")
    size_bytes: int = Field(alias="sizeBytes")

    model_config = ConfigDict(populate_by_name=True)


class MediaOptimizationCompleted(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: str = Field(default="media.optimization.completed", alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")

    source_event_id: str = Field(alias="sourceEventId")
    media_id: str = Field(alias="mediaId")

    canonical: OptimizedObject | None = None
    proxy: OptimizedObject | None = None
    thumbnail: OptimizedObject | None = None

    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = Field(default=None, alias="frameRate")
    codec: str | None = None

    raw_bucket_name: str = Field(alias="rawBucketName")
    raw_object_key: str = Field(alias="rawObjectKey")
    raw_size_bytes: int = Field(alias="rawSizeBytes")

    model_config = ConfigDict(populate_by_name=True)


class MediaOptimizationFailed(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: str = Field(default="media.optimization.failed", alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")

    source_event_id: str = Field(alias="sourceEventId")
    media_id: str = Field(alias="mediaId")

    error_code: str = Field(alias="errorCode")
    error_message: str = Field(alias="errorMessage")

    model_config = ConfigDict(populate_by_name=True)
