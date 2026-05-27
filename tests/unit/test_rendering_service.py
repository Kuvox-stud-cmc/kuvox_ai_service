from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kuvox_ai.modules.rendering import RenderingService
from kuvox_ai.modules.rendering.models import RenderJob
from kuvox_ai.schemas import Plan


async def test_render_not_implemented(mock_storage: AsyncMock) -> None:
    svc = RenderingService(storage=mock_storage)
    job = RenderJob(job_id=uuid4(), plan=Plan(), output_storage_key="out/x.mp4")
    with pytest.raises(NotImplementedError):
        await svc.render(job)
