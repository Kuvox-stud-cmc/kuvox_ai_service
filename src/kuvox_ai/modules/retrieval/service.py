"""Retrieval service — graph-augmented multimodal search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from qdrant_client import models

from kuvox_ai.infrastructure import KuzuClient, QdrantClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.text_encoder import (
    SentenceTransformerTextEncoder,
    TextEmbeddingEncoder,
)
from kuvox_ai.modules.retrieval.models import (
    RetrievalEvidenceSnippet,
    RetrievalQuery,
    VideoEditorRetrievalQuery,
    VideoEditorRetrievalResult,
    VideoEditorShotResult,
)
from kuvox_ai.schemas import RetrievalResult

logger = get_logger(__name__)

_SEARCHABLE_MODALITIES: dict[str, str] = {
    "transcript": "shots_transcript",
    "ocr": "shots_ocr",
}
_UNSUPPORTED_MODALITY_WARNINGS = {
    "visual": "Visual semantic retrieval is not available in V-012.",
    "audio": "Audio semantic retrieval is not available in V-012.",
}
_RRF_K = 60
_MAX_EVIDENCE_PER_SHOT = 4
_T = TypeVar("_T")


@dataclass(slots=True)
class _ShotAccumulator:
    shot_id: str
    media_id: str
    start_seconds: float
    end_seconds: float
    score: float = 0.0
    modality_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[RetrievalEvidenceSnippet] = field(default_factory=list)
    previous_shot_id: str | None = None
    next_shot_id: str | None = None


class RetrievalService:
    """Public interface to the retrieval pipeline.

    For each requested modality, queries the corresponding Qdrant collection,
    optionally expands the hit set through Kuzu graph traversal, then fuses
    per-modality rankings via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        *,
        kuzu: KuzuClient,
        qdrant: QdrantClient,
        text_encoder: TextEmbeddingEncoder | None = None,
        transcript_collection_name: str = "shots_transcript",
        ocr_collection_name: str = "shots_ocr",
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_embedding_device: str = "auto",
        text_embedding_batch_size: int = 32,
    ) -> None:
        self._kuzu = kuzu
        self._qdrant = qdrant
        self._text_encoder = text_encoder or SentenceTransformerTextEncoder(
            model_name=text_embedding_model_name,
            device=text_embedding_device,
            batch_size=text_embedding_batch_size,
        )
        self._collections = {
            "transcript": transcript_collection_name,
            "ocr": ocr_collection_name,
        }

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run the retrieval pipeline for a single query.

        Legacy endpoint compatibility. Editor retrieval requires trusted media
        ids, so the old generic query returns an empty result instead of
        issuing unrestricted vector searches.
        """
        logger.info(
            "retrieval.retrieve.start", has_query=bool(query.text.strip()), top_k=query.top_k
        )
        return RetrievalResult(query=query.text)

    async def retrieve_video_editor(
        self,
        query: VideoEditorRetrievalQuery,
    ) -> VideoEditorRetrievalResult:
        """Search transcript/OCR shot indexes for trusted editor media ids."""
        warnings: list[str] = []
        normalized_media_ids = _dedupe_non_empty(query.media_ids)
        normalized_modalities = _dedupe_non_empty(query.modalities)
        searchable_modalities: list[Literal["transcript", "ocr"]] = []
        if "transcript" in normalized_modalities:
            searchable_modalities.append("transcript")
        if "ocr" in normalized_modalities:
            searchable_modalities.append("ocr")

        for modality in normalized_modalities:
            warning = _UNSUPPORTED_MODALITY_WARNINGS.get(modality)
            if warning:
                warnings.append(warning)

        if not normalized_media_ids:
            warnings.append("No ready project video media is available for semantic retrieval.")
            return VideoEditorRetrievalResult(
                project_id=query.project_id,
                query=query.query,
                warnings=warnings,
            )

        if not query.query.strip():
            return VideoEditorRetrievalResult(
                project_id=query.project_id,
                query=query.query,
                warnings=warnings,
            )

        if not searchable_modalities:
            warnings.append("No V-012 searchable modalities were requested.")
            return VideoEditorRetrievalResult(
                project_id=query.project_id,
                query=query.query,
                warnings=warnings,
            )

        vectors = await self._text_encoder.encode_texts([query.query])
        if not vectors:
            return VideoEditorRetrievalResult(
                project_id=query.project_id,
                query=query.query,
                warnings=[*warnings, "The retrieval query could not be embedded."],
            )
        vector = vectors[0]

        accumulators: dict[str, _ShotAccumulator] = {}
        candidates_considered = 0

        for modality in searchable_modalities:
            collection_name = self._collections[modality]
            hits = await self._search_collection(
                collection_name=collection_name,
                modality=modality,
                vector=vector,
                media_ids=normalized_media_ids,
                limit=query.top_k,
                warnings=warnings,
            )
            candidates_considered += len(hits)
            for rank, hit in enumerate(hits, start=1):
                _merge_hit(accumulators, modality=modality, hit=hit, rank=rank)

        if query.expand_graph and accumulators:
            await self._attach_graph_context(accumulators, warnings)

        results = [
            VideoEditorShotResult(
                shot_id=shot.shot_id,
                media_id=shot.media_id,
                start_seconds=shot.start_seconds,
                end_seconds=shot.end_seconds,
                score=shot.score,
                modality_scores=shot.modality_scores,
                previous_shot_id=shot.previous_shot_id,
                next_shot_id=shot.next_shot_id,
                evidence=shot.evidence[:_MAX_EVIDENCE_PER_SHOT],
            )
            for shot in sorted(
                accumulators.values(),
                key=lambda item: (-item.score, item.start_seconds, item.shot_id),
            )[: query.top_k]
        ]

        return VideoEditorRetrievalResult(
            project_id=query.project_id,
            query=query.query,
            results=results,
            warnings=_dedupe_non_empty(warnings),
            total_candidates_considered=candidates_considered,
        )

    async def _search_collection(
        self,
        *,
        collection_name: str,
        modality: Literal["transcript", "ocr"],
        vector: list[float],
        media_ids: list[str],
        limit: int,
        warnings: list[str],
    ) -> list[Any]:
        try:
            client = self._qdrant.client
            if not await client.collection_exists(collection_name):
                warnings.append(f"Qdrant collection {collection_name} is not available.")
                return []

            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="mediaId",
                        match=models.MatchAny(any=media_ids),
                    )
                ]
            )
            if hasattr(client, "query_points"):
                response = await client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )
                return list(getattr(response, "points", response) or [])

            return list(
                await client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "retrieval.video_editor.collection_failed",
                collection=collection_name,
                modality=modality,
                error=str(exc),
            )
            warnings.append(f"Retrieval collection {collection_name} is unavailable.")
            return []

    async def _attach_graph_context(
        self,
        accumulators: dict[str, _ShotAccumulator],
        warnings: list[str],
    ) -> None:
        try:
            for shot in accumulators.values():
                context = await self._graph_neighbors(shot.shot_id)
                shot.previous_shot_id = context.get("previousShotId")
                shot.next_shot_id = context.get("nextShotId")
        except Exception as exc:  # noqa: BLE001
            logger.warning("retrieval.video_editor.graph_failed", error=str(exc))
            warnings.append("Kuzu graph expansion is unavailable.")

    async def _graph_neighbors(self, shot_id: str) -> dict[str, str | None]:
        result = await self._kuzu.execute(
            """
            MATCH (s:Shot {shot_id: $shot_id})
            OPTIONAL MATCH (prev:Shot)-[:NEXT]->(s)
            OPTIONAL MATCH (s)-[:NEXT]->(next:Shot)
            RETURN prev.shot_id AS previousShotId, next.shot_id AS nextShotId
            """,
            {"shot_id": shot_id},
        )
        row = _first_kuzu_row(result)
        return {
            "previousShotId": _string_or_none(_row_value(row, "previousShotId", 0)),
            "nextShotId": _string_or_none(_row_value(row, "nextShotId", 1)),
        }


def _merge_hit(
    accumulators: dict[str, _ShotAccumulator],
    *,
    modality: Literal["transcript", "ocr"],
    hit: Any,
    rank: int,
) -> None:
    payload = getattr(hit, "payload", None) or {}
    if not isinstance(payload, dict):
        return

    shot_id = _string_or_none(payload.get("shotId"))
    media_id = _string_or_none(payload.get("mediaId"))
    start_seconds = _float_or_none(payload.get("startSeconds"))
    end_seconds = _float_or_none(payload.get("endSeconds"))
    if not shot_id or not media_id or start_seconds is None or end_seconds is None:
        return

    raw_score = _float_or_none(getattr(hit, "score", None)) or 0.0
    fused_score = 1.0 / (_RRF_K + rank)
    accumulator = accumulators.get(shot_id)
    if accumulator is None:
        accumulator = _ShotAccumulator(
            shot_id=shot_id,
            media_id=media_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        accumulators[shot_id] = accumulator

    accumulator.score += fused_score
    accumulator.modality_scores[modality] = max(
        accumulator.modality_scores.get(modality, 0.0),
        raw_score,
    )
    text = _string_or_none(payload.get("text"))
    if text:
        accumulator.evidence.append(
            RetrievalEvidenceSnippet(
                modality=modality,
                text=_snippet(text),
                score=raw_score,
            )
        )


def _first_kuzu_row(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "has_next") and hasattr(result, "get_next"):
        return result.get_next() if result.has_next() else None
    if isinstance(result, list):
        return result[0] if result else None
    return None


def _row_value(row: Any, key: str, index: int) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, (list, tuple)):
        return row[index] if index < len(row) else None
    return getattr(row, key, None)


def _dedupe_non_empty(values: list[_T]) -> list[_T]:
    return list(dict.fromkeys(value for value in values if value))


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _snippet(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized[:280]
