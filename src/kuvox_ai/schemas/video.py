"""Video domain model."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Video(BaseModel):
    """A source video uploaded to the system."""

    id: UUID
    owner_id: UUID
    title: str
    storage_key: str = Field(description="Object-storage key for the source file.")
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    created_at: datetime
    # TODO: extend with codec, bitrate, language, processing status, etc.
