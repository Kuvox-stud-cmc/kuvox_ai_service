from __future__ import annotations

from uuid import uuid4

import pytest

from kuvox_ai.modules.sandbox import SandboxService
from kuvox_ai.modules.sandbox.models import SandboxJob


async def test_execute_not_implemented() -> None:
    svc = SandboxService()
    job = SandboxJob(job_id=uuid4(), code="print('hi')", timeout_seconds=5)
    with pytest.raises(NotImplementedError):
        await svc.execute(job)
