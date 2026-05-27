"""Integration test: requires `make up` services running.

Boots the real FastAPI app, hits /health, expects every dependency healthy.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kuvox_ai.main import create_app


@pytest.mark.integration
def test_health_endpoint_reports_all_healthy() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "healthy"
    names = {dep["name"] for dep in body["dependencies"]}
    assert names == {"kuzu", "qdrant", "redis", "rabbitmq", "object_storage", "llm"}
    for dep in body["dependencies"]:
        assert dep["status"] == "healthy", dep
