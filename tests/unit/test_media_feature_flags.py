from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kuvox_ai.api.routes.retrieval import router as retrieval_router
from kuvox_ai.api.state import get_state
from kuvox_ai.config import Settings, get_settings
from kuvox_ai.main import _start_workers, lifespan


def _settings(
    *,
    media_ingestion_enabled: bool = False,
    media_retrieval_enabled: bool = False,
    run_workers: bool = True,
) -> Settings:
    return Settings(
        s3_access_key="test",
        s3_secret_key="test",
        media_ingestion_enabled=media_ingestion_enabled,
        media_retrieval_enabled=media_retrieval_enabled,
        run_workers=run_workers,
    )


def test_media_features_default_off_and_require_explicit_true() -> None:
    defaults = _settings()
    enabled = _settings(media_ingestion_enabled=True, media_retrieval_enabled=True)
    non_explicit = Settings(
        media_ingestion_enabled="1",
        media_retrieval_enabled="yes",
        s3_access_key="test",
        s3_secret_key="test",
    )

    assert defaults.media_ingestion_enabled is False
    assert defaults.media_retrieval_enabled is False
    assert enabled.media_ingestion_enabled is True
    assert enabled.media_retrieval_enabled is True
    assert non_explicit.media_ingestion_enabled is False
    assert non_explicit.media_retrieval_enabled is False


def test_disabled_retrieval_returns_stable_503_before_state_or_service_access() -> None:
    app = FastAPI()
    app.include_router(retrieval_router)
    app.dependency_overrides[get_settings] = lambda: _settings(media_retrieval_enabled=False)
    app.dependency_overrides[get_state] = lambda: (_ for _ in ()).throw(
        AssertionError("retrieval state must not be resolved")
    )

    response = TestClient(app).post(
        "/retrieval/video-editor",
        json={
            "projectId": "project-1",
            "mediaIds": ["media-1"],
            "query": "find a beach",
            "modalities": ["transcript"],
            "topK": 4,
            "expandGraph": True,
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Media retrieval is disabled."}


@pytest.mark.asyncio
async def test_disabled_ingestion_keeps_optimization_rendering_and_sandbox_workers() -> None:
    rabbitmq = AsyncMock()
    state = SimpleNamespace(
        rabbitmq=rabbitmq,
        media_optimization=object(),
        ingestion=object(),
        rendering=object(),
    )

    await _start_workers(state, _settings(media_ingestion_enabled=False))  # type: ignore[arg-type]

    bound_queues = [call.args[0] for call in rabbitmq.consume_bound_queue.await_args_list]
    topology_queues = [call.args[0] for call in rabbitmq.declare_retry_topology.await_args_list]
    assert bound_queues == ["media.optimization.requested", "kuvox.rendering"]
    assert topology_queues == ["media.optimization.requested", "kuvox.rendering"]
    rabbitmq.consume.assert_awaited_once()
    assert rabbitmq.consume.await_args.args[0] == "kuvox.sandbox"


@pytest.mark.asyncio
async def test_lifespan_does_not_connect_kuzu_or_qdrant_when_semantic_features_are_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = {
        name: SimpleNamespace(connect=AsyncMock(), close=AsyncMock())
        for name in ("kuzu", "qdrant", "redis", "rabbitmq", "storage", "llm")
    }
    state = SimpleNamespace(**clients)
    settings = _settings(
        media_ingestion_enabled=False,
        media_retrieval_enabled=False,
        run_workers=False,
    )
    monkeypatch.setattr("kuvox_ai.main.get_settings", lambda: settings)
    monkeypatch.setattr("kuvox_ai.main._build_state", lambda _settings: state)
    app = FastAPI()

    async with lifespan(app):
        assert app.state.kuvox is state

    clients["kuzu"].connect.assert_not_awaited()
    clients["qdrant"].connect.assert_not_awaited()
    clients["kuzu"].close.assert_not_awaited()
    clients["qdrant"].close.assert_not_awaited()
    clients["rabbitmq"].connect.assert_awaited_once()
    clients["storage"].connect.assert_awaited_once()
