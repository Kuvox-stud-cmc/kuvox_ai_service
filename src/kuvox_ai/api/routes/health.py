"""Health-check router."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Response, status

from kuvox_ai.api.schemas import DependencyHealth, HealthResponse
from kuvox_ai.api.state import AppState, get_state

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, state: AppState = Depends(get_state)) -> HealthResponse:
    """Ping every infrastructure dependency in parallel."""
    checks = {
        "kuzu": state.kuzu.health_check(),
        "qdrant": state.qdrant.health_check(),
        "redis": state.redis.health_check(),
        "rabbitmq": state.rabbitmq.health_check(),
        "object_storage": state.storage.health_check(),
        "llm": state.llm.health_check(),
    }
    results = await asyncio.gather(*checks.values(), return_exceptions=True)

    deps: list[DependencyHealth] = []
    all_healthy = True
    for name, ok in zip(checks.keys(), results, strict=True):
        healthy = isinstance(ok, bool) and ok
        if not healthy:
            all_healthy = False
        deps.append(DependencyHealth(name=name, status="healthy" if healthy else "unhealthy"))

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        dependencies=deps,
    )
