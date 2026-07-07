"""Retrieval router."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from kuvox_ai.api.schemas import (
    RetrievalHttpRequest,
    RetrievalHttpResponse,
    VideoEditorRetrievalHttpRequest,
    VideoEditorRetrievalHttpResponse,
)
from kuvox_ai.api.state import AppState, get_state
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.retrieval.models import RetrievalQuery, VideoEditorRetrievalQuery

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
logger = get_logger(__name__)


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


@router.post("/video-editor", response_model=VideoEditorRetrievalHttpResponse)
async def retrieve_video_editor(
    body: VideoEditorRetrievalHttpRequest,
    state: AppState = Depends(get_state),
) -> VideoEditorRetrievalHttpResponse:
    """Run trusted, evidence-backed retrieval for the video editor."""
    logger.info(
        "retrieval.video_editor.route.start",
        project_id=body.project_id,
        media_count=len(body.media_ids),
        top_k=body.top_k,
        modalities=body.modalities,
        expand_graph=body.expand_graph,
    )
    query = VideoEditorRetrievalQuery(
        project_id=body.project_id,
        media_ids=body.media_ids,
        query=body.query,
        modalities=body.modalities,
        top_k=body.top_k,
        expand_graph=body.expand_graph,
    )
    try:
        result = await state.retrieval.retrieve_video_editor(query)
    except Exception:
        logger.exception(
            "retrieval.video_editor.route.failure",
            project_id=body.project_id,
            media_count=len(body.media_ids),
            top_k=body.top_k,
            modalities=body.modalities,
        )
        raise

    logger.info(
        "retrieval.video_editor.route.success",
        project_id=result.project_id,
        result_count=len(result.results),
        warning_count=len(result.warnings),
        total_candidates_considered=result.total_candidates_considered,
        top_k=body.top_k,
        modalities=body.modalities,
    )
    return VideoEditorRetrievalHttpResponse(result=result)
