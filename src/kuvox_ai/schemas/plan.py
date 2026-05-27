"""Plan domain model — the output of the planning module."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from kuvox_ai.schemas.operation import Operation


class Plan(BaseModel):
    """A validated, executable sequence of operations.

    Produced by :mod:`kuvox_ai.modules.planning` and consumed by
    :mod:`kuvox_ai.modules.rendering`.
    """

    id: UUID = Field(default_factory=uuid4)
    video_id: UUID | None = Field(
        default=None,
        description="Optional anchor video; some plans operate across multiple sources.",
    )
    operations: list[Operation] = Field(default_factory=list)
    rationale: str | None = Field(
        default=None,
        description="Optional human-readable explanation of why this plan was chosen.",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
