"""Module-local pydantic models for ingestion."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

IngestionTier = Literal[0, 1, 2]

_DEFAULT_TIERS: list[IngestionTier] = [0, 1, 2]


class IngestionRequest(BaseModel):
    """Payload pulled off the ingestion RabbitMQ queue."""

    video_id: UUID
    source_storage_key: str
    tiers: list[IngestionTier] = Field(default_factory=lambda: list(_DEFAULT_TIERS))


class IngestionResult(BaseModel):
    """Outcome of an ingestion run for a single video."""

    video_id: UUID
    shots_detected: int = 0
    tiers_completed: list[IngestionTier] = Field(default_factory=list)
