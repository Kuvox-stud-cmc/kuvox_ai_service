from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from starlette.responses import Response

from kuvox_ai.api.routes.health import live, ready
from kuvox_ai.api.state import AppState
from kuvox_ai.config import Settings


def _state(**statuses: bool) -> SimpleNamespace:
    names = ("kuzu", "qdrant", "redis", "rabbitmq", "storage", "llm")
    values = {name: statuses.get(name, True) for name in names}
    return SimpleNamespace(
        **{
            name: SimpleNamespace(health_check=AsyncMock(return_value=value))
            for name, value in values.items()
        }
    )


def _settings(*, cache_enabled: bool) -> Settings:
    return Settings(
        cache_enabled=cache_enabled,
        s3_access_key="test",
        s3_secret_key="test",
    )


@pytest.mark.asyncio
async def test_live_is_process_only() -> None:
    assert await live() == {"status": "healthy", "service": "kuvox-ai"}


@pytest.mark.asyncio
async def test_ready_reports_redis_disabled_when_cache_is_off() -> None:
    response = Response()
    result = await ready(
        response, cast(AppState, _state(redis=False)), _settings(cache_enabled=False)
    )
    redis = next(dep for dep in result.dependencies if dep.name == "redis")
    assert response.status_code == 200
    assert result.status == "healthy"
    assert redis.status == "disabled"
    assert redis.required is False


@pytest.mark.asyncio
async def test_redis_failure_is_degraded_but_ready() -> None:
    response = Response()
    result = await ready(
        response, cast(AppState, _state(redis=False)), _settings(cache_enabled=True)
    )
    assert response.status_code == 200
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_required_failure_is_unhealthy() -> None:
    response = Response()
    result = await ready(
        response, cast(AppState, _state(qdrant=False)), _settings(cache_enabled=True)
    )
    assert response.status_code == 503
    assert result.status == "unhealthy"


def test_metrics_route_is_registered_without_raw_identifier_labels() -> None:
    from prometheus_client import generate_latest

    from kuvox_ai.main import create_app

    paths = {path for route in create_app().routes if (path := getattr(route, "path", None))}
    assert "/metrics" in paths
    exposition = generate_latest().decode("utf-8")
    assert "kuvox_cache_operations_total" in exposition
    for forbidden in ("user_id", "studio_id", "project_id", "media_id", "query", "token"):
        assert f'{forbidden}="' not in exposition
