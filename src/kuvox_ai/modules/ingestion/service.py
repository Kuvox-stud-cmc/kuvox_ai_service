"""Ingestion service — orchestrates tiered video processing."""

from __future__ import annotations

from kuvox_ai.infrastructure import KuzuClient, ObjectStorageClient, QdrantClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.models import IngestionRequest, IngestionResult

logger = get_logger(__name__)


class IngestionService:
    """Public interface to the ingestion pipeline.

    Tier 0 — shot boundary detection, thumbnails, metadata.
    Tier 1 — visual embeddings (CLIP), speech transcription + embeddings (Whisper).
    Tier 2 — audio embeddings (CLAP), OCR, structured features.

    Writes graph data to Kuzu, vectors to Qdrant, media to object storage.
    """

    def __init__(
        self,
        *,
        kuzu: KuzuClient,
        qdrant: QdrantClient,
        storage: ObjectStorageClient,
    ) -> None:
        self._kuzu = kuzu
        self._qdrant = qdrant
        self._storage = storage

    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        """Run the configured tiers for a single video.

        TODO: implement the tiered pipeline. ML loading and processing live
        in dedicated sub-packages (e.g. ``tier0/``, ``tier1/``, ``tier2/``)
        added later — this scaffold intentionally stops at the entry point.
        """
        logger.info("ingestion.ingest.start", video_id=str(request.video_id))
        raise NotImplementedError("IngestionService.ingest is not implemented yet")
