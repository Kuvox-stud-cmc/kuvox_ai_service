from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kuvox_ai.api.middleware import RequestLoggingMiddleware
from kuvox_ai.api.routes.planning import router as planning_router
from kuvox_ai.api.routes.retrieval import router as retrieval_router
from kuvox_ai.api.state import get_state
from kuvox_ai.modules.planning.models import MoveItemAction, VideoEditorPlanningResponse
from kuvox_ai.modules.retrieval.models import (
    RetrievalEvidenceSnippet,
    VideoEditorRetrievalResult,
    VideoEditorShotResult,
)


def test_video_editor_routes_keep_camel_case_schema_and_echo_correlation_header() -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(planning_router)
    app.include_router(retrieval_router)
    app.dependency_overrides[get_state] = lambda: _State()
    client = TestClient(app)

    planning = client.post(
        "/planning/video-editor",
        headers={"x-kuvox-editor-correlation-id": "corr-test"},
        json=_planning_payload(),
    )

    assert planning.status_code == 200
    assert planning.headers["x-kuvox-editor-correlation-id"] == "corr-test"
    planning_body = planning.json()
    assert planning_body["commandId"] == "command-test"
    assert planning_body["actions"][0]["type"] == "moveItem"
    assert planning_body["actions"][0]["itemId"] == "tl-beach"
    assert planning_body["actions"][0]["timelineStart"] == 8
    assert "command_id" not in planning_body

    retrieval = client.post(
        "/retrieval/video-editor",
        headers={"x-kuvox-editor-correlation-id": "corr-test"},
        json={
            "projectId": "video-ai",
            "mediaIds": ["clip-beach"],
            "query": "b-roll",
            "modalities": ["transcript", "ocr"],
            "topK": 4,
            "expandGraph": True,
        },
    )

    assert retrieval.status_code == 200
    assert retrieval.headers["x-kuvox-editor-correlation-id"] == "corr-test"
    result = retrieval.json()["result"]
    assert result["projectId"] == "video-ai"
    assert result["totalCandidatesConsidered"] == 1
    assert result["results"][0]["shotId"] == "shot-1"
    assert result["results"][0]["startSeconds"] == 1
    assert result["results"][0]["evidence"][0]["modality"] == "transcript"


class _Planning:
    async def plan_video_editor(self, body):
        return VideoEditorPlanningResponse(
            ok=True,
            plan_id="plan-test",
            command_id=body.command_id,
            explanation="Moved clip.",
            confidence=0.9,
            actions=[MoveItemAction(item_id="tl-beach", timeline_start=8)],
            warnings=[],
        )


class _Retrieval:
    async def retrieve_video_editor(self, query):
        return VideoEditorRetrievalResult(
            project_id=query.project_id,
            query=query.query,
            total_candidates_considered=1,
            warnings=[],
            results=[
                VideoEditorShotResult(
                    shot_id="shot-1",
                    media_id=query.media_ids[0],
                    start_seconds=1,
                    end_seconds=4,
                    score=0.95,
                    modality_scores={"transcript": 0.95},
                    evidence=[
                        RetrievalEvidenceSnippet(
                            modality="transcript",
                            text="usable b-roll",
                            score=0.95,
                        )
                    ],
                )
            ],
        )


class _State:
    planning = _Planning()
    retrieval = _Retrieval()


def _planning_payload() -> dict[str, object]:
    return {
        "projectId": "video-ai",
        "commandId": "command-test",
        "command": "move clip 1 to 8s",
        "playheadTime": 6,
        "selection": {"selectedItemIds": ["tl-beach"], "activeItemId": "tl-beach"},
        "playback": {"currentTime": 6},
        "timeline": {
            "projectId": "video-ai",
            "revision": 1,
            "frameRate": 30,
            "media": [{"id": "clip-beach", "kind": "video", "name": "Beach", "duration": 24}],
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
                }
            ],
        },
        "availableMedia": [],
    }
