"""Planning router."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from kuvox_ai.api.schemas import (
    PlanningHttpRequest,
    PlanningHttpResponse,
    VideoEditorPlanningHttpRequest,
    VideoEditorPlanningHttpResponse,
)
from kuvox_ai.api.state import AppState, get_state
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.planning.models import PlanningRequest

router = APIRouter(prefix="/planning", tags=["planning"])
logger = get_logger(__name__)


@router.post("", response_model=PlanningHttpResponse)
async def plan(
    body: PlanningHttpRequest,
    state: AppState = Depends(get_state),
) -> PlanningHttpResponse:
    """Produce a validated Plan for the given natural-language command."""
    request = PlanningRequest(
        command=body.command,
        video_id=body.video_id,
        context_shot_ids=body.context_shot_ids,
    )
    result = await state.planning.plan(request)
    return PlanningHttpResponse(plan=result)


@router.post("/video-editor", response_model=VideoEditorPlanningHttpResponse)
async def plan_video_editor(
    body: VideoEditorPlanningHttpRequest,
    state: AppState = Depends(get_state),
) -> VideoEditorPlanningHttpResponse:
    """Produce deterministic video-editor actions for concrete commands."""
    logger.info(
        "planning.video_editor.route.start",
        project_id=body.project_id,
        command_id=body.command_id,
        selected_item_count=len(body.selection.selected_item_ids),
        track_count=len(body.timeline.tracks),
    )
    try:
        result = await state.planning.plan_video_editor(body)
    except Exception:
        logger.exception(
            "planning.video_editor.route.failure",
            project_id=body.project_id,
            command_id=body.command_id,
        )
        raise

    logger.info(
        "planning.video_editor.route.success",
        project_id=body.project_id,
        command_id=result.command_id,
        plan_id=result.plan_id,
        ok=result.ok,
        action_count=len(result.actions),
        warning_count=len(result.warnings),
    )
    return VideoEditorPlanningHttpResponse.model_validate(result.model_dump())
