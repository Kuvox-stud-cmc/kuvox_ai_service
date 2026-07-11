"""Renderer manifest schema for editor-export compatibility checks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def require_visual_dimensions(self) -> VideoRenderMediaSource:
        if self.kind != "audio" and (self.width is None or self.height is None):
            raise ValueError("Renderable visual media sources require width and height.")
        return self


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


class VideoRenderKeyframe(_CamelModel):
    time: float = Field(ge=0)
    value: float
    easing: tuple[float, float, float, float] | None = None


class VideoRenderAnimationTrack(_CamelModel):
    keyframes: list[VideoRenderKeyframe] = Field(min_length=1)


class VideoRenderTransformAnimation(_CamelModel):
    x: VideoRenderAnimationTrack | None = None
    y: VideoRenderAnimationTrack | None = None
    scale_x: VideoRenderAnimationTrack | None = None
    scale_y: VideoRenderAnimationTrack | None = None
    rotation: VideoRenderAnimationTrack | None = None


class VideoRenderCropAnimation(_CamelModel):
    top: VideoRenderAnimationTrack | None = None
    right: VideoRenderAnimationTrack | None = None
    bottom: VideoRenderAnimationTrack | None = None
    left: VideoRenderAnimationTrack | None = None


class VideoRenderAnimation(_CamelModel):
    transform: VideoRenderTransformAnimation | None = None
    crop: VideoRenderCropAnimation | None = None
    opacity: VideoRenderAnimationTrack | None = None


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
    stack_order: int
    transform: VideoRenderTransform
    crop: VideoRenderCrop
    opacity: float = Field(ge=0, le=1)
    animation: VideoRenderAnimation | None = None


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
    stroke_color: str | None = None
    stroke_width: float | None = Field(default=None, ge=0)
    shadow_color: str | None = None
    shadow_blur: float | None = Field(default=None, ge=0)
    shadow_offset_x: float | None = None
    shadow_offset_y: float | None = None
    anim_type: str | None = None
    anim_dur: float | None = Field(default=None, ge=0)


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
    stack_order: int
    animation: VideoRenderAnimation | None = None


class VideoRenderManifest(_CamelModel):
    schema_version: Literal[1, 2]
    project_id: str
    settings: VideoRenderSettings
    duration_seconds: float = Field(ge=0)
    media_sources: list[VideoRenderMediaSource] = Field(default_factory=list)
    visual_items: list[VideoRenderVisualItem] = Field(default_factory=list)
    audio_items: list[VideoRenderAudioItem] = Field(default_factory=list)
    text_overlays: list[VideoRenderTextOverlay] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def upgrade_v1_stack_order(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("schemaVersion", value.get("schema_version")) != 1:
            return value
        upgraded = dict(value)
        visuals = [dict(item) for item in upgraded.get("visualItems", [])]
        texts = [dict(item) for item in upgraded.get("textOverlays", [])]
        ordered = sorted(
            [("visual", index, item) for index, item in enumerate(visuals)]
            + [("text", index, item) for index, item in enumerate(texts)],
            key=lambda entry: (
                float(entry[2].get("timelineStart", 0)),
                int(entry[2].get("layerOrder", 0)),
                str(entry[2].get("itemId", "")),
            ),
        )
        for stack_order, (kind, index, _) in enumerate(ordered):
            target = visuals if kind == "visual" else texts
            target[index]["stackOrder"] = stack_order
            if kind == "visual" and "crop" not in target[index]:
                target[index]["crop"] = {"top": 0, "right": 0, "bottom": 0, "left": 0}
        upgraded["visualItems"] = visuals
        upgraded["textOverlays"] = texts
        return upgraded
