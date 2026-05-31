"""Schemas for /retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kuvox_ai.schemas import RetrievalResult

Modality = Literal["visual", "transcript", "audio", "ocr"]

_DEFAULT_MODALITIES: list[Modality] = ["visual", "transcript", "audio", "ocr"]


class RetrievalHttpRequest(BaseModel):
    query: str
    modalities: list[Modality] = Field(default_factory=lambda: list(_DEFAULT_MODALITIES))
    top_k: int = Field(default=20, ge=1, le=200)
    expand_graph: bool = True


class RetrievalHttpResponse(BaseModel):
    result: RetrievalResult
