from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kuvox_ai.modules.rendering import RenderingService
from kuvox_ai.modules.rendering.models import RenderJob


async def test_render_not_implemented(mock_storage: AsyncMock) -> None:
    svc = RenderingService(storage=mock_storage)
    job = RenderJob.model_validate(
        {
            "eventId": "evt-1",
            "eventType": "rendering.requested",
            "occurredAt": "2026-07-09T00:00:00Z",
            "renderJobId": "job-1",
            "timelineId": "timeline-1",
            "projectId": "project-1",
            "revisionId": "revision-1",
            "revisionNumber": 1,
            "requestedByUserId": "user-1",
            "settings": {"format": "mp4"},
            "documentJson": {"tracks": [], "media": {}},
            "mediaSources": [],
            "outputBucketName": "kuvox-renders",
            "outputStorageKey": "out/x.mp4",
            "outputContentType": "video/mp4",
        }
    )
    with pytest.raises(NotImplementedError):
        await svc.render(job)
