"""Schemas for /retrieval."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kuvox_ai.modules.retrieval.models import VideoEditorRetrievalResult
from kuvox_ai.schemas import RetrievalResult

Modality = Literal["visual", "transcript", "audio", "ocr"]

_DEFAULT_MODALITIES: list[Modality] = ["visual", "transcript", "audio", "ocr"]


def _default_editor_modalities() -> list[Modality]:
    return ["transcript", "ocr"]


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


class RetrievalHttpRequest(BaseModel):
    query: str
    modalities: list[Modality] = Field(default_factory=lambda: list(_DEFAULT_MODALITIES))
    top_k: int = Field(default=20, ge=1, le=200)
    expand_graph: bool = True


class RetrievalHttpResponse(BaseModel):
    result: RetrievalResult


class VideoEditorRetrievalHttpRequest(_CamelModel):
    project_id: str
    media_ids: list[str] = Field(default_factory=list)
    query: str
    modalities: list[Modality] = Field(default_factory=_default_editor_modalities)
    top_k: int = Field(default=8, ge=1, le=50)
    expand_graph: bool = True
    scope_revision: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_scope_revisions(cls, value: Any) -> Any:
        if (
            isinstance(value, dict)
            and "scopeRevision" in value
            and "scope_revision" in value
            and value["scopeRevision"] != value["scope_revision"]
        ):
            value = dict(value)
            value["scopeRevision"] = None
            value.pop("scope_revision", None)
        return value


class VideoEditorRetrievalHttpResponse(_CamelModel):
    result: VideoEditorRetrievalResult
