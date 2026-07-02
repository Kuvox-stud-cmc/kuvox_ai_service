"""Ingestion service — orchestrates tiered video processing."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from kuvox_ai.infrastructure import KuzuClient, ObjectStorageClient, QdrantClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.kuzu_writer import KuzuIngestionWriter
from kuvox_ai.modules.ingestion.metadata import probe_video_metadata
from kuvox_ai.modules.ingestion.models import (
    IngestionCompleted,
    IngestionRequested,
    MediaKind,
    VideoMetadata,
)
from kuvox_ai.modules.ingestion.shot_detection import detect_video_shots

logger = get_logger(__name__)


class _LegacyIngestionService:
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

    async def ingest(self, request: Any) -> object:
        """Run the configured tiers for a single video.

        TODO: implement the tiered pipeline. ML loading and processing live
        in dedicated sub-packages (e.g. ``tier0/``, ``tier1/``, ``tier2/``)
        added later — this scaffold intentionally stops at the entry point.
        """
        logger.info("ingestion.ingest.start", video_id=str(request.video_id))
        raise NotImplementedError("IngestionService.ingest is not implemented yet")


class IngestionService:
    """Download canonical video, detect shots, and write Video/Shot graph data."""

    def __init__(
        self,
        *,
        kuzu: KuzuClient,
        storage: ObjectStorageClient,
        work_dir: Path,
        qdrant: QdrantClient | None = None,
        writer: KuzuIngestionWriter | None = None,
    ) -> None:
        self._kuzu = kuzu
        self._storage = storage
        self._work_dir = work_dir
        self._qdrant = qdrant
        self._writer = writer or KuzuIngestionWriter(kuzu)

    async def ingest(self, request: IngestionRequested) -> IngestionCompleted:
        if request.kind != MediaKind.video:
            raise ValueError(f"Ingestion MVP 1 only supports video media, got {request.kind}.")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ingestion.ingest.start", media_id=request.media_id)

        with temporary_job_dir(self._work_dir) as job_dir:
            canonical_path = job_dir / canonical_filename(request.canonical.object_key)
            await self._storage.download_file(
                request.canonical.bucket_name,
                request.canonical.object_key,
                canonical_path,
            )

            metadata = merge_metadata(
                request_metadata=VideoMetadata(
                    duration_seconds=request.duration_seconds,
                    width=request.width,
                    height=request.height,
                    frame_rate=request.frame_rate,
                    codec=request.codec,
                ),
                probed_metadata=await probe_video_metadata(canonical_path),
            )
            shots = await detect_video_shots(
                canonical_path,
                media_id=request.media_id,
                duration_seconds=metadata.duration_seconds or 0.0,
            )
            await self._writer.write_video_with_shots(request, metadata, shots)

        logger.info("ingestion.ingest.completed", media_id=request.media_id, shot_count=len(shots))
        return IngestionCompleted(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            shot_count=len(shots),
        )


def canonical_filename(object_key: str) -> str:
    return Path(object_key).name or "canonical.mp4"


@contextmanager
def temporary_job_dir(work_dir: Path) -> Iterator[Path]:
    job_dir = work_dir / f"job-{uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield job_dir
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def merge_metadata(
    *,
    request_metadata: VideoMetadata,
    probed_metadata: VideoMetadata,
) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=probed_metadata.duration_seconds or request_metadata.duration_seconds,
        width=probed_metadata.width or request_metadata.width,
        height=probed_metadata.height or request_metadata.height,
        frame_rate=probed_metadata.frame_rate or request_metadata.frame_rate,
        codec=probed_metadata.codec or request_metadata.codec,
    )
