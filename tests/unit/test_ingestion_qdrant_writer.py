from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client import models

from kuvox_ai.infrastructure import QdrantClient
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot, IngestionRequested
from kuvox_ai.modules.ingestion.qdrant_writer import (
    QdrantShotEmbeddingWriter,
    QdrantShotVisualWriter,
    ShotVectorPoint,
)


def request() -> IngestionRequested:
    return IngestionRequested(
        event_id="evt-1",
        event_type="ingestion.requested",
        occurred_at=datetime(2026, 7, 2, tzinfo=UTC),
        media_id="media-1",
        owner_id="owner-1",
        owner_kind="User",
        kind="Video",
        canonical={
            "bucketName": "kuvox-canonical",
            "objectKey": "media/media-1/canonical.mp4",
            "contentType": "video/mp4",
            "sizeBytes": 1,
        },
    )


def sampled_frame() -> SampledFrame:
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=1.0,
        end_seconds=5.0,
        duration_seconds=4.0,
    )
    return SampledFrame(shot=shot, timestamp_seconds=3.0, path=Path("shot.jpg"))


class FakeQdrantWrapper:
    def __init__(self, client: Any) -> None:
        self.client = client


class FakeQdrantNativeClient:
    def __init__(self, *, collection_exists: bool, vector_size: int = 512) -> None:
        self._collection_exists = collection_exists
        self._vector_size = vector_size
        self.created: list[dict[str, object]] = []
        self.deleted: list[dict[str, object]] = []
        self.upserted: list[dict[str, object]] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return self._collection_exists

    async def get_collection(self, collection_name: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=self._vector_size,
                        distance=models.Distance.COSINE,
                    )
                )
            )
        )

    async def create_collection(self, **kwargs: object) -> None:
        self.created.append(kwargs)

    async def delete(self, **kwargs: object) -> None:
        self.deleted.append(kwargs)

    async def upsert(self, **kwargs: object) -> None:
        self.upserted.append(kwargs)


async def test_qdrant_writer_creates_collection_deletes_media_and_upserts_points() -> None:
    native = FakeQdrantNativeClient(collection_exists=False)
    writer = QdrantShotVisualWriter(
        cast(QdrantClient, FakeQdrantWrapper(native)),
        collection_name="shots_visual",
        embedding_dim=2,
    )

    await writer.write_shot_vectors(request(), [sampled_frame()], [[0.1, 0.2]])

    assert native.created[0]["collection_name"] == "shots_visual"
    vectors_config = native.created[0]["vectors_config"]
    assert isinstance(vectors_config, models.VectorParams)
    assert vectors_config.size == 2
    assert vectors_config.distance == models.Distance.COSINE
    assert native.deleted[0]["collection_name"] == "shots_visual"
    selector = native.deleted[0]["points_selector"]
    assert isinstance(selector, models.FilterSelector)
    assert selector.filter.must[0].key == "mediaId"
    assert native.upserted[0]["collection_name"] == "shots_visual"
    points = native.upserted[0]["points"]
    assert isinstance(points, list)
    assert points[0].id == "media-1:shot:000000"
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {
        "mediaId": "media-1",
        "ownerId": "owner-1",
        "ownerKind": "User",
        "shotId": "media-1:shot:000000",
        "shotIndex": 0,
        "startSeconds": 1.0,
        "endSeconds": 5.0,
        "durationSeconds": 4.0,
        "frameTimestampSeconds": 3.0,
    }


async def test_qdrant_writer_rejects_incompatible_collection_vector_size() -> None:
    native = FakeQdrantNativeClient(collection_exists=True, vector_size=768)
    writer = QdrantShotVisualWriter(
        cast(QdrantClient, FakeQdrantWrapper(native)),
        collection_name="shots_visual",
        embedding_dim=512,
    )

    with pytest.raises(ValueError, match="vector size 768"):
        await writer.write_shot_vectors(request(), [sampled_frame()], [[0.0] * 512])

    assert native.deleted == []
    assert native.upserted == []


async def test_qdrant_writer_rejects_wrong_embedding_dimensions() -> None:
    native = FakeQdrantNativeClient(collection_exists=False)
    writer = QdrantShotVisualWriter(
        cast(QdrantClient, FakeQdrantWrapper(native)),
        collection_name="shots_visual",
        embedding_dim=512,
    )

    with pytest.raises(ValueError, match="dimension 2"):
        await writer.write_shot_vectors(request(), [sampled_frame()], [[0.1, 0.2]])

    assert native.created == []
    assert native.deleted == []
    assert native.upserted == []


async def test_generic_qdrant_writer_deletes_and_skips_empty_upsert() -> None:
    native = FakeQdrantNativeClient(collection_exists=False)
    writer = QdrantShotEmbeddingWriter(
        cast(QdrantClient, FakeQdrantWrapper(native)),
        collection_name="shots_transcript",
        embedding_dim=384,
    )

    await writer.write_points(request(), [])

    assert native.created[0]["collection_name"] == "shots_transcript"
    assert native.deleted[0]["collection_name"] == "shots_transcript"
    assert native.upserted == []


async def test_generic_qdrant_writer_upserts_points() -> None:
    native = FakeQdrantNativeClient(collection_exists=True, vector_size=384)
    writer = QdrantShotEmbeddingWriter(
        cast(QdrantClient, FakeQdrantWrapper(native)),
        collection_name="shots_transcript",
        embedding_dim=384,
    )

    await writer.write_points(
        request(),
        [
            ShotVectorPoint(
                point_id="media-1:shot:000000",
                vector=[0.1] * 384,
                payload={"mediaId": "media-1", "modality": "transcript"},
            )
        ],
    )

    assert native.created == []
    assert native.deleted[0]["collection_name"] == "shots_transcript"
    points = native.upserted[0]["points"]
    assert isinstance(points, list)
    assert points[0].id == "media-1:shot:000000"
    assert points[0].payload == {"mediaId": "media-1", "modality": "transcript"}
