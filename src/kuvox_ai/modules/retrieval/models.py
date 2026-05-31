"""Module-local pydantic models for retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["visual", "transcript", "audio", "ocr"]

_DEFAULT_MODALITIES: list[Modality] = ["visual", "transcript", "audio", "ocr"]


class RetrievalQuery(BaseModel):
    """Input to :meth:`RetrievalService.retrieve`."""

    text: str
    modalities: list[Modality] = Field(default_factory=lambda: list(_DEFAULT_MODALITIES))
    top_k: int = Field(default=20, ge=1, le=200)
    expand_graph: bool = Field(
        default=True,
        description="If true, expand initial vector hits through Kuzu graph traversal.",
    )
