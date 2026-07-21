from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kuvox_ai.modules.rendering.adjustments import REGISTRY_VERSION, resolve_visual_style


def test_adjustment_registry_matches_shared_typescript_fixtures() -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "video-adjustments-v1.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixtures["registryVersion"] == REGISTRY_VERSION
    for fixture in fixtures["cases"]:
        item = dict(fixture["item"])
        assert resolve_visual_style(item) == fixture["expected"], fixture["id"]

    frontend_fixture = (
        Path(__file__).parents[3]
        / "kuvox_frontend"
        / "scripts"
        / "fixtures"
        / "video-adjustments-v1.json"
    )
    if frontend_fixture.exists():
        frontend = json.loads(frontend_fixture.read_text(encoding="utf-8"))
        assert frontend == fixtures


def test_adjustment_registry_output_is_json_serializable() -> None:
    style: dict[str, Any] = resolve_visual_style({"type": "video", "properties": {}})
    assert json.loads(json.dumps(style))["filter"]["brightness"] == 1
