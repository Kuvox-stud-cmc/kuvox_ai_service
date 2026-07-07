"""Renderer manifest schema for editor-export compatibility checks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True, extra="forbid")


class VideoRenderSettings(_CamelModel):
    preset: Literal["h264-720p", "h264-1080p", "h264-4k", "prores-master"]
    format: Literal["mp4", "mov"]
    resolution: Literal["1280x720", "1920x1080", "3840x2160", "current"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: Literal[24, 25, 30, 60]
    quality: Literal["draft", "standard", "high"]
    destination_label: str


class VideoRenderCanonicalSource(_CamelModel):
    variant: Literal["canonical"]
    url: str
    storage_key: str


class VideoRenderMediaSource(_CamelModel):
    media_id: str
    kind: Literal["video", "audio", "image"]
    name: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str | None = None
    canonical: VideoRenderCanonicalSource


class VideoRenderTransform(_CamelModel):
    x: float
    y: float
    scale_x: float = Field(gt=0)
    scale_y: float = Field(gt=0)
    rotation: float


class VideoRenderCrop(_CamelModel):
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)
    left: float = Field(ge=0, le=1)


class VideoRenderVisualItem(_CamelModel):
    item_id: str
    track_id: str
    type: Literal["video", "image", "overlay"]
    media_id: str
    shot_id: str | None = None
    timeline_start: float = Field(ge=0)
    duration: float = Field(gt=0)
    source_in: float | None = Field(default=None, ge=0)
    source_out: float | None = Field(default=None, gt=0)
    speed: float | None = Field(default=None, gt=0)
    layer_order: int
    transform: VideoRenderTransform
    crop: VideoRenderCrop | None = None
    opacity: float = Field(ge=0, le=1)


class VideoRenderAudioFades(_CamelModel):
    fade_in_duration: float = Field(ge=0)
    fade_out_duration: float = Field(ge=0)


class VideoRenderAudioItem(_CamelModel):
    item_id: str
    track_id: str
    media_id: str
    timeline_start: float = Field(ge=0)
    duration: float = Field(gt=0)
    source_in: float = Field(ge=0)
    source_out: float = Field(gt=0)
    speed: float = Field(gt=0)
    volume: float = Field(ge=0, le=1)
    muted: bool
    fades: VideoRenderAudioFades
    layer_order: int


class VideoRenderTextStyle(_CamelModel):
    font_family: str
    font_size: float = Field(gt=0)
    color: str
    background_color: str | None = None
    font_weight: Literal["normal", "medium", "semibold", "bold"] | None = None
    font_style: Literal["normal", "italic"] | None = None
    text_align: Literal["left", "center", "right"] | None = None


class VideoRenderTextOverlay(_CamelModel):
    item_id: str
    track_id: str
    text: str
    timeline_start: float = Field(ge=0)
    duration: float = Field(gt=0)
    style: VideoRenderTextStyle
    transform: VideoRenderTransform
    opacity: float = Field(ge=0, le=1)
    layer_order: int


class VideoRenderManifest(_CamelModel):
    schema_version: Literal[1]
    project_id: str
    settings: VideoRenderSettings
    duration_seconds: float = Field(ge=0)
    media_sources: list[VideoRenderMediaSource] = Field(default_factory=list)
    visual_items: list[VideoRenderVisualItem] = Field(default_factory=list)
    audio_items: list[VideoRenderAudioItem] = Field(default_factory=list)
    text_overlays: list[VideoRenderTextOverlay] = Field(default_factory=list)
