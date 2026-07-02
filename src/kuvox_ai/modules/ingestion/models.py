"""Message and domain models for ingestion MVP 1."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MediaKind(StrEnum):
    video = "Video"
    audio = "Audio"
    image = "Image"


class OwnerKind(StrEnum):
    user = "User"
    studio = "Studio"


class OptimizedObject(BaseModel):
    bucket_name: str = Field(alias="bucketName")
    object_key: str = Field(alias="objectKey")
    content_type: str = Field(alias="contentType")
    size_bytes: int = Field(alias="sizeBytes")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class IngestionRequested(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: Literal["ingestion.requested"] = Field(alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")

    media_id: str = Field(alias="mediaId")
    owner_id: str = Field(alias="ownerId")
    owner_kind: OwnerKind = Field(alias="ownerKind")
    kind: MediaKind

    canonical: OptimizedObject
    proxy: OptimizedObject | None = None
    thumbnail: OptimizedObject | None = None

    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = Field(default=None, alias="frameRate")
    codec: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class VideoMetadata(BaseModel):
    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = Field(default=None, alias="frameRate")
    codec: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class DetectedShot(BaseModel):
    shot_id: str = Field(alias="shotId")
    media_id: str = Field(alias="mediaId")
    shot_index: int = Field(alias="shotIndex")
    start_seconds: float = Field(alias="startSeconds")
    end_seconds: float = Field(alias="endSeconds")
    duration_seconds: float = Field(alias="durationSeconds")

    model_config = ConfigDict(populate_by_name=True)


class IngestionCompleted(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: Literal["ingestion.completed"] = Field(
        default="ingestion.completed",
        alias="eventType",
    )
    occurred_at: datetime = Field(alias="occurredAt")
    source_event_id: str = Field(alias="sourceEventId")
    media_id: str = Field(alias="mediaId")
    shot_count: int = Field(alias="shotCount")

    model_config = ConfigDict(populate_by_name=True)


class IngestionFailed(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: Literal["ingestion.failed"] = Field(
        default="ingestion.failed",
        alias="eventType",
    )
    occurred_at: datetime = Field(alias="occurredAt")
    source_event_id: str = Field(alias="sourceEventId")
    media_id: str = Field(alias="mediaId")
    error_code: str = Field(alias="errorCode")
    error_message: str = Field(alias="errorMessage")

    model_config = ConfigDict(populate_by_name=True)


def shot_id_for(media_id: str, shot_index: int) -> str:
    return f"{media_id}:shot:{shot_index:06d}"
