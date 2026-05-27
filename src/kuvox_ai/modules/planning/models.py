"""Module-local pydantic models for planning."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

PlanningPath = Literal["direct", "reasoning", "code_generation"]


class PlanningRequest(BaseModel):
    """Input to :meth:`PlanningService.plan`."""

    command: str = Field(description="The user's natural-language editing command.")
    video_id: UUID | None = None
    context_shot_ids: list[UUID] = Field(default_factory=list)


class PlanningTrace(BaseModel):
    """Diagnostic trace of how a plan was produced (returned alongside the plan)."""

    path: PlanningPath
    notes: list[str] = Field(default_factory=list)
