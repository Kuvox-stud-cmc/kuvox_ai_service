from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from qdrant_client import models

from kuvox_ai.infrastructure import KuzuClient, QdrantClient
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.modules.retrieval import RetrievalService
from kuvox_ai.modules.retrieval.models import VideoEditorRetrievalQuery


class FakeTextEncoder:
    def __init__(self) -> None:
        self.texts: list[list[str]] = []

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.append(texts)
        return [[0.1, 0.2, 0.3]]


class FakeQdrantWrapper:
    def __init__(self, client: Any) -> None:
        self.client = client


class FakeQdrantNativeClient:
    def __init__(
        self,
        *,
        collections: set[str] | None = None,
        hits_by_collection: dict[str, list[Any]] | None = None,
        fail_collections: set[str] | None = None,
    ) -> None:
        self.collections = collections or {"shots_transcript", "shots_ocr"}
        self.hits_by_collection = hits_by_collection or {}
        self.fail_collections = fail_collections or set()
        self.queries: list[dict[str, Any]] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def query_points(self, **kwargs: Any) -> Any:
        collection_name = str(kwargs["collection_name"])
        if collection_name in self.fail_collections:
            raise RuntimeError("collection unavailable")
        self.queries.append(kwargs)
        return SimpleNamespace(points=self.hits_by_collection.get(collection_name, []))


class FakeKuzu:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[dict[str, Any]] = []

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        self.queries.append({"query": query, "params": params})
        if self.fail:
            raise RuntimeError("graph unavailable")
        shot_id = params["shot_id"] if params else "shot"
        return [{"previousShotId": f"{shot_id}:prev", "nextShotId": f"{shot_id}:next"}]


async def test_video_editor_retrieval_empty_trusted_media_does_not_query_qdrant() -> None:
    qdrant = FakeQdrantNativeClient()
    encoder = FakeTextEncoder()
    svc = service(qdrant=qdrant, encoder=encoder)

    result = await svc.retrieve_video_editor(
        VideoEditorRetrievalQuery(
            projectId="project-1",
            mediaIds=[],
            query="sunset",
        )
    )

    assert result.results == []
    assert qdrant.queries == []
    assert encoder.texts == []
    assert result.warnings == ["No ready project video media is available for semantic retrieval."]


async def test_video_editor_retrieval_filters_to_trusted_media_ids() -> None:
    qdrant = FakeQdrantNativeClient(
        hits_by_collection={
            "shots_transcript": [
                hit(
                    "media-1:shot:000001",
                    "media-1",
                    2.0,
                    5.0,
                    0.91,
                    text="the founder speaks on stage",
                )
            ],
            "shots_ocr": [],
        }
    )
    svc = service(qdrant=qdrant)

    result = await svc.retrieve_video_editor(
        VideoEditorRetrievalQuery(
            projectId="project-1",
            mediaIds=["media-1", "media-2"],
            query="founder on stage",
            modalities=["transcript"],
            topK=5,
            expandGraph=False,
        )
    )

    assert [item.shot_id for item in result.results] == ["media-1:shot:000001"]
    assert result.results[0].media_id == "media-1"
    assert result.results[0].evidence[0].text == "the founder speaks on stage"
    assert len(qdrant.queries) == 1
    query_filter = qdrant.queries[0]["query_filter"]
    assert isinstance(query_filter, models.Filter)
    match = query_filter.must[0].match
    assert isinstance(match, models.MatchAny)
    assert match.any == ["media-1", "media-2"]


async def test_video_editor_retrieval_fuses_duplicate_shots_and_preserves_evidence() -> None:
    qdrant = FakeQdrantNativeClient(
        hits_by_collection={
            "shots_transcript": [
                hit("shot-1", "media-1", 0.0, 4.0, 0.8, text="transcript evidence"),
                hit("shot-2", "media-1", 4.0, 8.0, 0.7, text="second evidence"),
            ],
            "shots_ocr": [
                hit("shot-1", "media-1", 0.0, 4.0, 0.6, text="ocr evidence"),
            ],
        }
    )
    kuzu = FakeKuzu()
    svc = service(qdrant=qdrant, kuzu=kuzu)

    result = await svc.retrieve_video_editor(
        VideoEditorRetrievalQuery(
            projectId="project-1",
            mediaIds=["media-1"],
            query="evidence",
            modalities=["transcript", "ocr"],
            topK=5,
            expandGraph=True,
        )
    )

    assert [item.shot_id for item in result.results] == ["shot-1", "shot-2"]
    first = result.results[0]
    assert first.modality_scores == {"transcript": 0.8, "ocr": 0.6}
    assert [snippet.modality for snippet in first.evidence] == ["transcript", "ocr"]
    assert first.previous_shot_id == "shot-1:prev"
    assert first.next_shot_id == "shot-1:next"
    assert result.total_candidates_considered == 3


async def test_video_editor_retrieval_returns_warnings_for_degraded_dependencies() -> None:
    qdrant = FakeQdrantNativeClient(
        collections={"shots_transcript"},
        hits_by_collection={
            "shots_transcript": [hit("shot-1", "media-1", 0.0, 4.0, 0.8, text="partial")],
        },
    )
    svc = service(qdrant=qdrant, kuzu=FakeKuzu(fail=True))

    result = await svc.retrieve_video_editor(
        VideoEditorRetrievalQuery(
            projectId="project-1",
            mediaIds=["media-1"],
            query="partial",
            modalities=["visual", "transcript", "ocr", "audio"],
            expandGraph=True,
        )
    )

    assert [item.shot_id for item in result.results] == ["shot-1"]
    assert "Visual semantic retrieval is not available in V-012." in result.warnings
    assert "Audio semantic retrieval is not available in V-012." in result.warnings
    assert "Qdrant collection shots_ocr is not available." in result.warnings
    assert "Kuzu graph expansion is unavailable." in result.warnings


def service(
    *,
    qdrant: FakeQdrantNativeClient,
    kuzu: FakeKuzu | None = None,
    encoder: FakeTextEncoder | None = None,
) -> RetrievalService:
    return RetrievalService(
        kuzu=cast(KuzuClient, kuzu or FakeKuzu()),
        qdrant=cast(QdrantClient, FakeQdrantWrapper(qdrant)),
        text_encoder=cast(TextEmbeddingEncoder, encoder or FakeTextEncoder()),
    )


def hit(
    shot_id: str,
    media_id: str,
    start_seconds: float,
    end_seconds: float,
    score: float,
    *,
    text: str,
) -> Any:
    return SimpleNamespace(
        score=score,
        payload={
            "shotId": shot_id,
            "mediaId": media_id,
            "startSeconds": start_seconds,
            "endSeconds": end_seconds,
            "text": text,
        },
    )
