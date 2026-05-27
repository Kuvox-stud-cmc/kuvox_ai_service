"""Module-local pydantic models for rendering."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from kuvox_ai.schemas import Plan


class RenderJob(BaseModel):
    """Payload pulled off the rendering RabbitMQ queue."""

    job_id: UUID
    plan: Plan
    output_format: Literal["mp4", "webm", "mov"] = "mp4"
    output_storage_key: str = Field(description="Object-storage key for the rendered output.")


class RenderResult(BaseModel):
    """Outcome of a render job."""

    job_id: UUID
    output_storage_key: str
    duration_seconds: float
