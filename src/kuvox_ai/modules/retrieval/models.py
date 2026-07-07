"""Module-local pydantic models for retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Modality = Literal["visual", "transcript", "audio", "ocr"]

_DEFAULT_MODALITIES: list[Modality] = ["visual", "transcript", "audio", "ocr"]


def _default_editor_modalities() -> list[Modality]:
    return ["transcript", "ocr"]


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


class RetrievalQuery(BaseModel):
    """Input to :meth:`RetrievalService.retrieve`."""

    text: str
    modalities: list[Modality] = Field(default_factory=lambda: list(_DEFAULT_MODALITIES))
    top_k: int = Field(default=20, ge=1, le=200)
    expand_graph: bool = Field(
        default=True,
        description="If true, expand initial vector hits through Kuzu graph traversal.",
    )


class VideoEditorRetrievalQuery(_CamelModel):
    """Trusted editor retrieval input derived at the BFF/API boundary."""

    project_id: str
    media_ids: list[str] = Field(default_factory=list)
    query: str
    modalities: list[Modality] = Field(default_factory=_default_editor_modalities)
    top_k: int = Field(default=8, ge=1, le=50)
    expand_graph: bool = True


class RetrievalEvidenceSnippet(_CamelModel):
    modality: Literal["transcript", "ocr"]
    text: str
    score: float


class VideoEditorShotResult(_CamelModel):
    shot_id: str
    media_id: str
    start_seconds: float
    end_seconds: float
    score: float
    modality_scores: dict[str, float] = Field(default_factory=dict)
    previous_shot_id: str | None = None
    next_shot_id: str | None = None
    evidence: list[RetrievalEvidenceSnippet] = Field(default_factory=list)


class VideoEditorRetrievalResult(_CamelModel):
    project_id: str
    query: str
    results: list[VideoEditorShotResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_candidates_considered: int = 0
