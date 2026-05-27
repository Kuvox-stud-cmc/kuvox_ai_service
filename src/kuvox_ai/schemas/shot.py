"""Shot domain model — a contiguous segment of a video between cuts."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class Shot(BaseModel):
    """A single shot within a :class:`Video`."""

    id: UUID
    video_id: UUID
    index: int = Field(description="Zero-based index of the shot within the video.")
    start_seconds: float
    end_seconds: float
    thumbnail_key: str | None = Field(
        default=None, description="Object-storage key for the shot thumbnail."
    )
    transcript: str | None = None
    # TODO: extend with detected entities, OCR text, dominant colors, etc.

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds
