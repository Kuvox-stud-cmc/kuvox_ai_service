"""Schemas for /planning."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from kuvox_ai.modules.planning.models import (
    VideoEditorPlanningRequest,
    VideoEditorPlanningResponse,
)
from kuvox_ai.schemas import Plan


class PlanningHttpRequest(BaseModel):
    command: str
    video_id: UUID | None = None
    context_shot_ids: list[UUID] = []


class PlanningHttpResponse(BaseModel):
    plan: Plan


class VideoEditorPlanningHttpRequest(VideoEditorPlanningRequest):
    pass


class VideoEditorPlanningHttpResponse(VideoEditorPlanningResponse):
    pass
