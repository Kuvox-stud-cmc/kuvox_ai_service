"""Round-trip tests for cross-module schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from kuvox_ai.schemas import (
    ConcatenateOperation,
    Plan,
    RetrievalResult,
    ScoredShot,
    Shot,
    TransitionOperation,
    TrimOperation,
    Video,
)


def test_video_round_trip() -> None:
    v = Video(
        id=uuid4(),
        owner_id=uuid4(),
        title="t",
        storage_key="videos/x.mp4",
        created_at=datetime.utcnow(),
    )
    assert Video.model_validate_json(v.model_dump_json()) == v


def test_shot_duration() -> None:
    s = Shot(id=uuid4(), video_id=uuid4(), index=0, start_seconds=1.0, end_seconds=4.5)
    assert s.duration_seconds == pytest.approx(3.5)


def test_operation_discriminated_union() -> None:
    shot_id = uuid4()
    trim = TrimOperation(shot_id=shot_id, start_seconds=0.0, end_seconds=2.0)
    concat = ConcatenateOperation(shot_ids=[shot_id, uuid4()])
    transition = TransitionOperation(from_shot_id=shot_id, to_shot_id=uuid4())

    plan = Plan(operations=[trim, concat, transition])
    reparsed = Plan.model_validate_json(plan.model_dump_json())
    assert [op.op for op in reparsed.operations] == ["trim", "concatenate", "transition"]


def test_retrieval_result_defaults() -> None:
    r = RetrievalResult(query="a dog running")
    assert r.shots == []
    r2 = RetrievalResult(
        query="x",
        shots=[
            ScoredShot(
                shot=Shot(id=uuid4(), video_id=uuid4(), index=0, start_seconds=0, end_seconds=1),
                score=0.9,
                modality_scores={"visual": 0.9},
            )
        ],
    )
    assert r2.shots[0].score == 0.9
