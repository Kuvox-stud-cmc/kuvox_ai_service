"""Rendering service — executes Plans into finished videos."""

from __future__ import annotations

from kuvox_ai.infrastructure import ObjectStorageClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.rendering.models import RenderJob, RenderResult

logger = get_logger(__name__)


class RenderingService:
    """Public interface to the rendering pipeline.

    Walks the :class:`Plan`'s operations, drives MoviePy / FFmpeg to produce
    the output file, and uploads it to object storage.
    """

    def __init__(self, *, storage: ObjectStorageClient) -> None:
        self._storage = storage

    async def render(self, job: RenderJob) -> RenderResult:
        """Execute one render job end-to-end.

        TODO: implement MoviePy/FFmpeg dispatch per operation type, with
        temp-file management and upload of the final output.
        """
        logger.info(
            "rendering.render.start",
            render_job_id=job.render_job_id,
            n_operations=len(job.plan.operations),
            n_media_sources=len(job.media_sources),
        )
        raise NotImplementedError("RenderingService.render is not implemented yet")
