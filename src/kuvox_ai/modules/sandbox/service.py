"""Sandbox service — runs untrusted Python in isolated Docker containers."""

from __future__ import annotations

from kuvox_ai.logging import get_logger
from kuvox_ai.modules.sandbox.models import SandboxJob, SandboxResult

logger = get_logger(__name__)


class SandboxService:
    """Public interface to the sandbox executor.

    Isolation guarantees (enforced by the container runtime, not this service):

    * No outbound network.
    * Read-only mounts for any input files.
    * Hard CPU / memory / pids limits.
    * Hard wall-clock timeout (killed by the runtime, not the process).
    """

    def __init__(self, *, image: str = "kuvox-sandbox:latest") -> None:
        self._image = image

    async def execute(self, job: SandboxJob) -> SandboxResult:
        """Run one sandboxed job and return the captured outcome.

        TODO: implement Docker SDK invocation with the locked-down container
        config (network=none, read-only rootfs, no-new-privileges, etc.).
        """
        logger.info("sandbox.execute.start", job_id=str(job.job_id), timeout=job.timeout_seconds)
        raise NotImplementedError("SandboxService.execute is not implemented yet")
