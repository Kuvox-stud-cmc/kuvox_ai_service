"""Planning router."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from kuvox_ai.api.schemas import PlanningHttpRequest, PlanningHttpResponse
from kuvox_ai.api.state import AppState, get_state
from kuvox_ai.modules.planning.models import PlanningRequest

router = APIRouter(prefix="/planning", tags=["planning"])


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
