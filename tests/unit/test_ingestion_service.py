"""Ingestion service is constructed with mocked infrastructure and stubs out today."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kuvox_ai.modules.ingestion import IngestionService
from kuvox_ai.modules.ingestion.models import IngestionRequest


async def test_ingest_not_implemented(
    mock_kuzu: AsyncMock, mock_qdrant: AsyncMock, mock_storage: AsyncMock
) -> None:
    svc = IngestionService(kuzu=mock_kuzu, qdrant=mock_qdrant, storage=mock_storage)
    with pytest.raises(NotImplementedError):
        await svc.ingest(IngestionRequest(video_id=uuid4(), source_storage_key="videos/x.mp4"))
