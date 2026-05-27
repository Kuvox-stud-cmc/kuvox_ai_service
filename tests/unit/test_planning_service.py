from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kuvox_ai.infrastructure import StubLLMClient
from kuvox_ai.modules.planning import PlanningService
from kuvox_ai.modules.planning.models import PlanningRequest
from kuvox_ai.modules.retrieval import RetrievalService


async def test_plan_not_implemented(
    stub_llm: StubLLMClient, mock_kuzu: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    retrieval = RetrievalService(kuzu=mock_kuzu, qdrant=mock_qdrant)
    svc = PlanningService(llm=stub_llm, retrieval=retrieval)
    with pytest.raises(NotImplementedError):
        await svc.plan(PlanningRequest(command="make it dramatic"))
