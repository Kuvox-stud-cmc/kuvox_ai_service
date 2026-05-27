from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kuvox_ai.modules.retrieval import RetrievalService
from kuvox_ai.modules.retrieval.models import RetrievalQuery


async def test_retrieve_not_implemented(mock_kuzu: AsyncMock, mock_qdrant: AsyncMock) -> None:
    svc = RetrievalService(kuzu=mock_kuzu, qdrant=mock_qdrant)
    with pytest.raises(NotImplementedError):
        await svc.retrieve(RetrievalQuery(text="a dog"))
