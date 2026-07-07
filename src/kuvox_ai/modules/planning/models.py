"""Module-local pydantic models for planning."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

PlanningPath = Literal["direct", "reasoning", "code_generation"]


class PlanningRequest(BaseModel):
    """Input to :meth:`PlanningService.plan`."""

    command: str = Field(description="The user's natural-language editing command.")
    video_id: UUID | None = None
    context_shot_ids: list[UUID] = Field(default_factory=list)


class PlanningTrace(BaseModel):
    """Diagnostic trace of how a plan was produced (returned alongside the plan)."""

    path: PlanningPath
    notes: list[str] = Field(default_factory=list)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


class VideoEditorSelectionSummary(_CamelModel):
    selected_item_ids: list[str] = Field(default_factory=list)
    active_item_id: str | None = None


class VideoEditorPlaybackSummary(_CamelModel):
    current_time: float = 0.0


class VideoEditorMediaSummary(_CamelModel):
    id: str
    kind: Literal["video", "audio", "image"]
    name: str
    duration: float | None = None


class VideoEditorTimelineItemSummary(_CamelModel):
    id: str
    type: Literal["video", "audio", "text", "image", "overlay"]
    media_id: str | None = None
    shot_id: str | None = None
    timeline_start: float
    duration: float
    source_in: float | None = None
    source_out: float | None = None
    speed: float | None = None
    text: str | None = None


class VideoEditorTrackSummary(_CamelModel):
    id: str
    kind: Literal["video", "audio", "text", "overlay"]
    label: str
    locked: bool = False
    hidden: bool = False
    muted: bool = False
    items: list[VideoEditorTimelineItemSummary] = Field(default_factory=list)


class VideoEditorDocumentSummary(_CamelModel):
    project_id: str
    revision: int | None = None
    frame_rate: float = 30.0
    tracks: list[VideoEditorTrackSummary] = Field(default_factory=list)
    media: list[VideoEditorMediaSummary] = Field(default_factory=list)


class VideoEditorPlanningRequest(_CamelModel):
    project_id: str
    command_id: str
    command: str
    playhead_time: float = 0.0
    selection: VideoEditorSelectionSummary = Field(default_factory=VideoEditorSelectionSummary)
    playback: VideoEditorPlaybackSummary | None = None
    timeline: VideoEditorDocumentSummary
    available_media: list[VideoEditorMediaSummary] = Field(default_factory=list)


class _VideoEditorActionBase(_CamelModel):
    type: str


class TrimItemAction(_VideoEditorActionBase):
    type: Literal["trimItem"] = "trimItem"
    item_id: str
    timeline_start: float
    duration: float
    source_in: float | None = None
    source_out: float | None = None


class SplitItemAction(_VideoEditorActionBase):
    type: Literal["splitItem"] = "splitItem"
    item_id: str
    timeline_time: float


class DeleteItemsAction(_VideoEditorActionBase):
    type: Literal["deleteItems"] = "deleteItems"
    item_ids: list[str]


class MoveItemAction(_VideoEditorActionBase):
    type: Literal["moveItem"] = "moveItem"
    item_id: str
    timeline_start: float
    target_track_id: str | None = None


class AddTextAction(_VideoEditorActionBase):
    type: Literal["addText"] = "addText"
    text: str
    timeline_start: float
    track_id: str | None = None


class UpdateAudioAction(_VideoEditorActionBase):
    type: Literal["updateAudio"] = "updateAudio"
    item_id: str
    volume: float | None = None
    muted: bool | None = None


class UpdateSpeedAction(_VideoEditorActionBase):
    type: Literal["updateSpeed"] = "updateSpeed"
    item_id: str
    speed: float


VideoEditorPlanningAction = Annotated[
    TrimItemAction
    | SplitItemAction
    | DeleteItemsAction
    | MoveItemAction
    | AddTextAction
    | UpdateAudioAction
    | UpdateSpeedAction,
    Field(discriminator="type"),
]


class VideoEditorPlanningResponse(_CamelModel):
    ok: bool
    plan_id: str
    command_id: str
    actions: list[VideoEditorPlanningAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_reason: str | None = None
