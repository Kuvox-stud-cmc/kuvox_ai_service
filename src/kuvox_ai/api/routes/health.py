"""Health-check router."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Response, status

from kuvox_ai.api.schemas import DependencyHealth, HealthResponse
from kuvox_ai.api.state import AppState, get_state
from kuvox_ai.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Process-only liveness probe."""
    return {"status": "healthy", "service": "kuvox-ai"}


@router.get("/health", response_model=HealthResponse)
@router.get("/health/ready", response_model=HealthResponse)
async def ready(
    response: Response,
    state: AppState = Depends(get_state),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Check required dependencies and report optional Redis degradation."""
    semantic_dependencies_enabled = (
        settings.media_ingestion_enabled or settings.media_retrieval_enabled
    )
    checks = {
        "rabbitmq": state.rabbitmq.health_check(),
        "object_storage": state.storage.health_check(),
        "llm": state.llm.health_check(),
    }
    if semantic_dependencies_enabled:
        checks = {
            "kuzu": state.kuzu.health_check(),
            "qdrant": state.qdrant.health_check(),
            **checks,
        }
    if settings.cache_enabled:
        checks["redis"] = state.redis.health_check()
    results = await asyncio.gather(*checks.values(), return_exceptions=True)

    deps: list[DependencyHealth] = []
    required_unhealthy = False
    for name, ok in zip(checks.keys(), results, strict=True):
        healthy = isinstance(ok, bool) and ok
        required = name != "redis"
        if required and not healthy:
            required_unhealthy = True
        deps.append(
            DependencyHealth(
                name=name,
                status="healthy" if healthy else "unhealthy",
                required=required,
            )
        )

    if not settings.cache_enabled:
        deps.append(DependencyHealth(name="redis", status="disabled", required=False))
    if not semantic_dependencies_enabled:
        deps.extend(
            [
                DependencyHealth(name="kuzu", status="disabled", required=False),
                DependencyHealth(name="qdrant", status="disabled", required=False),
            ]
        )

    if required_unhealthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall = "unhealthy"
    elif any(dep.status == "unhealthy" for dep in deps):
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthResponse(
        status=overall,
        dependencies=deps,
    )
