"""Qdrant index writes for ingested shot embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import models

from kuvox_ai.infrastructure import QdrantClient
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import IngestionRequested


@dataclass(frozen=True, slots=True)
class ShotVectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, object]


class VisualIndexWriter(Protocol):
    async def write_shot_vectors(
        self,
        request: IngestionRequested,
        frames: list[SampledFrame],
        embeddings: list[list[float]],
    ) -> None: ...


class ShotEmbeddingIndexWriter(Protocol):
    async def write_points(
        self,
        request: IngestionRequested,
        points: list[ShotVectorPoint],
    ) -> None: ...


class QdrantShotEmbeddingWriter:
    def __init__(
        self,
        qdrant: QdrantClient,
        *,
        collection_name: str,
        embedding_dim: int,
    ) -> None:
        self._qdrant = qdrant
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

    async def write_points(
        self,
        request: IngestionRequested,
        points: list[ShotVectorPoint],
    ) -> None:
        for index, point in enumerate(points):
            if len(point.vector) != self._embedding_dim:
                raise ValueError(
                    f"Embedding {index} has dimension {len(point.vector)}, "
                    f"expected {self._embedding_dim}."
                )

        await self.ensure_collection()
        await self.delete_media_points(request.media_id)
        if not points:
            return

        points = [
            models.PointStruct(
                id=qdrant_point_id(point.point_id),
                vector=point.vector,
                payload=payload_with_logical_point_id(point),
            )
            for point in points
        ]
        await self._qdrant.client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    async def ensure_collection(self) -> None:
        client = self._qdrant.client
        if await client.collection_exists(self._collection_name):
            collection = await client.get_collection(self._collection_name)
            vector_size = collection_vector_size(collection)
            if vector_size != self._embedding_dim:
                raise ValueError(
                    f"Qdrant collection {self._collection_name!r} has vector size "
                    f"{vector_size}, expected {self._embedding_dim}."
                )
            distance = collection_vector_distance(collection)
            if distance is not None and distance != models.Distance.COSINE:
                raise ValueError(
                    f"Qdrant collection {self._collection_name!r} uses distance "
                    f"{distance}, expected {models.Distance.COSINE}."
                )
            return

        await client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )

    async def delete_media_points(self, media_id: str) -> None:
        await self._qdrant.client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="mediaId",
                            match=models.MatchValue(value=media_id),
                        )
                    ]
                )
            ),
            wait=True,
        )


class QdrantShotVisualWriter:
    def __init__(
        self,
        qdrant: QdrantClient,
        *,
        collection_name: str,
        embedding_dim: int,
    ) -> None:
        self._writer = QdrantShotEmbeddingWriter(
            qdrant,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
        )

    async def write_shot_vectors(
        self,
        request: IngestionRequested,
        frames: list[SampledFrame],
        embeddings: list[list[float]],
    ) -> None:
        if len(frames) != len(embeddings):
            raise ValueError(f"Expected {len(frames)} visual embeddings, got {len(embeddings)}.")

        points = [
            ShotVectorPoint(
                point_id=frame.shot.shot_id,
                vector=embedding,
                payload={
                    "mediaId": request.media_id,
                    "ownerId": request.owner_id,
                    "ownerKind": request.owner_kind.value,
                    "shotId": frame.shot.shot_id,
                    "shotIndex": frame.shot.shot_index,
                    "startSeconds": frame.shot.start_seconds,
                    "endSeconds": frame.shot.end_seconds,
                    "durationSeconds": frame.shot.duration_seconds,
                    "frameTimestampSeconds": frame.timestamp_seconds,
                },
            )
            for frame, embedding in zip(frames, embeddings, strict=True)
        ]
        await self._writer.write_points(request, points)


def collection_vector_size(collection: Any) -> int | None:
    vectors = collection_vector_config(collection)
    value = _field(vectors, "size")
    return int(value) if value is not None else None


def collection_vector_distance(collection: Any) -> models.Distance | None:
    vectors = collection_vector_config(collection)
    value = _field(vectors, "distance")
    if value is None:
        return None
    if isinstance(value, models.Distance):
        return value
    return models.Distance(str(value))


def collection_vector_config(collection: Any) -> Any:
    params = _field(_field(collection, "config"), "params")
    vectors = _field(params, "vectors")
    if isinstance(vectors, dict) and "size" not in vectors and vectors:
        first = next(iter(vectors.values()))
        return first
    return vectors


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def qdrant_point_id(logical_point_id: str) -> str:
    try:
        return str(UUID(logical_point_id))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"kuvox:qdrant:{logical_point_id}"))


def payload_with_logical_point_id(point: ShotVectorPoint) -> dict[str, object]:
    if "pointId" in point.payload:
        return point.payload
    return {**point.payload, "pointId": point.point_id}
