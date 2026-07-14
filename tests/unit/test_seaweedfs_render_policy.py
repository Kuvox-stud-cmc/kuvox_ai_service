from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_render_bucket_permissions_are_declared_for_worker_and_api() -> None:
    policy_path = Path(__file__).resolve().parents[3] / "infra" / "seaweedfs" / "s3.json.example"
    if not policy_path.is_file():
        pytest.skip("shared SeaweedFS policy is not available in this checkout")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    actions_by_identity = {
        identity["name"]: set(identity["actions"]) for identity in policy["identities"]
    }

    assert {
        "List:kuvox-renders",
        "Read:kuvox-renders",
        "Read:kuvox-renders/*",
        "Write:kuvox-renders",
        "Write:kuvox-renders/*",
    } <= actions_by_identity["kuvox-ai"]
    assert {
        "List:kuvox-renders",
        "Read:kuvox-renders",
        "Read:kuvox-renders/*",
    } <= actions_by_identity["kuvox-api"]
