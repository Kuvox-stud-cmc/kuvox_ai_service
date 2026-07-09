"""Module-local pydantic models for rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kuvox_ai.schemas import Plan


class RenderingMediaSource(BaseModel):
    media_id: str = Field(alias="mediaId")
    kind: str
    bucket_name: str | None = Field(default=None, alias="bucketName")
    object_key: str = Field(alias="objectKey")
    content_type: str | None = Field(default=None, alias="contentType")
    size_bytes: int = Field(alias="sizeBytes")
    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = Field(default=None, alias="frameRate")
    codec: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class RenderJob(BaseModel):
    """Payload pulled off the rendering RabbitMQ queue."""

    event_id: str = Field(alias="eventId")
    event_type: Literal["rendering.requested"] = Field(alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")

    render_job_id: str = Field(alias="renderJobId")
    timeline_id: str = Field(alias="timelineId")
    project_id: str = Field(alias="projectId")
    revision_id: str = Field(alias="revisionId")
    revision_number: int = Field(alias="revisionNumber")
    requested_by_user_id: str = Field(alias="requestedByUserId")

    settings: dict[str, Any]
    document_json: dict[str, Any] = Field(alias="documentJson")
    media_sources: list[RenderingMediaSource] = Field(default_factory=list, alias="mediaSources")
    output_bucket_name: str = Field(alias="outputBucketName")
    output_storage_key: str = Field(alias="outputStorageKey")
    output_content_type: str = Field(alias="outputContentType")

    plan: Plan = Field(default_factory=Plan)
    output_format: Literal["mp4", "webm", "mov"] = "mp4"

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RenderResult(BaseModel):
    """Outcome of a render job."""

    render_job_id: str = Field(alias="renderJobId")
    output_bucket_name: str | None = Field(default=None, alias="outputBucketName")
    output_storage_key: str
    output_content_type: str | None = Field(default=None, alias="outputContentType")
    output_size_bytes: int = Field(default=0, alias="outputSizeBytes")
    duration_seconds: float = Field(default=0, alias="durationSeconds")

    model_config = ConfigDict(populate_by_name=True)


class RenderingStarted(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: Literal["rendering.started"] = Field(default="rendering.started", alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")
    source_event_id: str = Field(alias="sourceEventId")
    render_job_id: str = Field(alias="renderJobId")
    started_at: datetime = Field(alias="startedAt")

    model_config = ConfigDict(populate_by_name=True)


class RenderingCompleted(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: Literal["rendering.completed"] = Field(
        default="rendering.completed", alias="eventType"
    )
    occurred_at: datetime = Field(alias="occurredAt")
    source_event_id: str = Field(alias="sourceEventId")
    render_job_id: str = Field(alias="renderJobId")
    output_bucket_name: str = Field(alias="outputBucketName")
    output_storage_key: str = Field(alias="outputStorageKey")
    output_content_type: str = Field(alias="outputContentType")
    output_size_bytes: int = Field(alias="outputSizeBytes")
    finished_at: datetime = Field(alias="finishedAt")

    model_config = ConfigDict(populate_by_name=True)


class RenderingFailed(BaseModel):
    event_id: str = Field(alias="eventId")
    event_type: Literal["rendering.failed"] = Field(default="rendering.failed", alias="eventType")
    occurred_at: datetime = Field(alias="occurredAt")
    source_event_id: str = Field(alias="sourceEventId")
    render_job_id: str = Field(alias="renderJobId")
    error_code: str = Field(alias="errorCode")
    error_message: str = Field(alias="errorMessage")
    finished_at: datetime = Field(alias="finishedAt")

    model_config = ConfigDict(populate_by_name=True)
