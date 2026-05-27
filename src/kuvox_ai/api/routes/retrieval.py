"""Retrieval router."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from kuvox_ai.api.schemas import RetrievalHttpRequest, RetrievalHttpResponse
from kuvox_ai.api.state import AppState, get_state
from kuvox_ai.modules.retrieval.models import RetrievalQuery

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("", response_model=RetrievalHttpResponse)
async def retrieve(
    body: RetrievalHttpRequest,
    state: AppState = Depends(get_state),
) -> RetrievalHttpResponse:
    """Run a graph-augmented multimodal retrieval query."""
    query = RetrievalQuery(
        text=body.query,
        modalities=body.modalities,
        top_k=body.top_k,
        expand_graph=body.expand_graph,
    )
    result = await state.retrieval.retrieve(query)
    return RetrievalHttpResponse(result=result)
