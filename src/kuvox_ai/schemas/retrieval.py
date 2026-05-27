"""Retrieval result domain model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kuvox_ai.schemas.shot import Shot


class ScoredShot(BaseModel):
    """A shot with a fused relevance score and per-modality breakdown."""

    shot: Shot
    score: float = Field(description="Fused score (e.g. Reciprocal Rank Fusion output).")
    modality_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-modality scores keyed by modality name (visual, transcript, audio, ocr).",
    )


class RetrievalResult(BaseModel):
    """Ranked shot candidates returned from a retrieval query."""

    query: str
    shots: list[ScoredShot] = Field(default_factory=list)
    total_candidates_considered: int = 0
