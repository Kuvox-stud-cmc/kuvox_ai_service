"""Administrative endpoints (re-ingestion, queue inspection, etc.).

All handlers are intentionally stubbed — wire up real admin tooling here.
The ASP.NET layer is responsible for gating who can call these.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from kuvox_ai.api.state import AppState, get_state

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reindex", status_code=202)
async def reindex(_state: AppState = Depends(get_state)) -> dict[str, str]:
    """Re-run ingestion for every video in the catalog.

    TODO: enumerate videos and dispatch ingestion jobs.
    """
    raise NotImplementedError("admin.reindex is not implemented yet")


@router.get("/queues")
async def queues(_state: AppState = Depends(get_state)) -> dict[str, int]:
    """Return current depth of each managed queue.

    TODO: call into RabbitMQ to fetch message counts.
    """
    raise NotImplementedError("admin.queues is not implemented yet")
