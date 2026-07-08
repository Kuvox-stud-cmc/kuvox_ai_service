from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kuvox_ai.api.routes.planning import router as planning_router
from kuvox_ai.api.state import get_state
from kuvox_ai.infrastructure import StubLLMClient
from kuvox_ai.modules.planning import PlanningService
from kuvox_ai.modules.planning.models import PlanningRequest, VideoEditorPlanningRequest
from kuvox_ai.modules.retrieval import RetrievalService


async def test_plan_not_implemented(
    stub_llm: StubLLMClient, mock_kuzu: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    retrieval = RetrievalService(kuzu=mock_kuzu, qdrant=mock_qdrant)
    svc = PlanningService(llm=stub_llm, retrieval=retrieval)
    with pytest.raises(NotImplementedError):
        await svc.plan(PlanningRequest(command="make it dramatic"))


async def test_video_editor_planning_handles_concrete_commands_without_retrieval(
    stub_llm: StubLLMClient, mock_kuzu: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    retrieval = RetrievalService(kuzu=mock_kuzu, qdrant=mock_qdrant)
    svc = PlanningService(llm=stub_llm, retrieval=retrieval)

    trim = await svc.plan_video_editor(_video_request("trim clip 1 from 4s to 10s"))
    assert trim.ok is True
    assert trim.command_id == "command-test"
    assert len(trim.actions) == 1
    action = trim.actions[0]
    assert action.type == "trimItem"
    assert action.item_id == "tl-beach"
    assert action.timeline_start == 4
    assert action.duration == 6
    assert action.source_in == 2
    assert action.source_out == 8

    split = await svc.plan_video_editor(_video_request("split here"))
    assert split.ok is True
    assert split.actions[0].type == "splitItem"
    assert split.actions[0].item_id == "tl-beach"
    assert split.actions[0].timeline_time == 6

    delete = await svc.plan_video_editor(
        _video_request(
            "delete",
            selection={
                "selectedItemIds": ["tl-caption", "tl-audio-main"],
                "activeItemId": "tl-caption",
            },
        )
    )
    assert delete.ok is True
    assert delete.actions[0].type == "deleteItems"
    assert delete.actions[0].item_ids == ["tl-caption", "tl-audio-main"]

    assert mock_kuzu.method_calls == []
    assert mock_qdrant.method_calls == []


async def test_video_editor_planning_returns_unsupported_for_abstract_commands(
    stub_llm: StubLLMClient, mock_kuzu: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    retrieval = RetrievalService(kuzu=mock_kuzu, qdrant=mock_qdrant)
    svc = PlanningService(llm=stub_llm, retrieval=retrieval)

    result = await svc.plan_video_editor(_video_request("make this cinematic"))

    assert result.ok is False
    assert result.actions == []
    assert result.unsupported_reason is not None


def test_video_editor_planning_route_returns_stable_camel_case_payload(
    stub_llm: StubLLMClient, mock_kuzu: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    retrieval = RetrievalService(kuzu=mock_kuzu, qdrant=mock_qdrant)
    svc = PlanningService(llm=stub_llm, retrieval=retrieval)
    app = FastAPI()
    app.include_router(planning_router)
    app.dependency_overrides[get_state] = lambda: type("State", (), {"planning": svc})()

    response = TestClient(app).post(
        "/planning/video-editor",
        json=_video_request_payload("move clip 1 to 12s"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["commandId"] == "command-test"
    assert body["actions"][0]["type"] == "moveItem"
    assert body["actions"][0]["itemId"] == "tl-beach"
    assert body["actions"][0]["timelineStart"] == 12


def _video_request(
    command: str,
    *,
    selection: dict[str, object] | None = None,
) -> VideoEditorPlanningRequest:
    return VideoEditorPlanningRequest.model_validate(
        _video_request_payload(command, selection=selection)
    )


def _video_request_payload(
    command: str,
    *,
    selection: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "projectId": "video-ai",
        "commandId": "command-test",
        "command": command,
        "playheadTime": 6,
        "selection": selection or {"selectedItemIds": ["tl-beach"], "activeItemId": "tl-beach"},
        "playback": {"currentTime": 6},
        "timeline": {
            "projectId": "video-ai",
            "revision": 1,
            "frameRate": 30,
            "media": [
                {"id": "clip-beach", "kind": "video", "name": "Beach", "duration": 24},
                {"id": "audio-main", "kind": "audio", "name": "Main audio", "duration": 30},
            ],
            "tracks": [
                {
                    "id": "v1",
                    "kind": "video",
                    "label": "Video 1",
                    "items": [
                        {
                            "id": "tl-beach",
                            "type": "video",
                            "mediaId": "clip-beach",
                            "timelineStart": 4,
                            "duration": 12,
                            "sourceIn": 2,
                            "sourceOut": 14,
                            "speed": 1,
                        }
                    ],
                },
                {
                    "id": "a1",
                    "kind": "audio",
                    "label": "Audio 1",
                    "items": [
                        {
                            "id": "tl-audio-main",
                            "type": "audio",
                            "mediaId": "audio-main",
                            "timelineStart": 0,
                            "duration": 20,
                            "sourceIn": 0,
                            "sourceOut": 20,
                        }
                    ],
                },
                {
                    "id": "t1",
                    "kind": "text",
                    "label": "Text",
                    "items": [
                        {
                            "id": "tl-caption",
                            "type": "text",
                            "timelineStart": 5,
                            "duration": 4,
                            "text": "Caption",
                        }
                    ],
                },
            ],
        },
        "availableMedia": [],
    }
