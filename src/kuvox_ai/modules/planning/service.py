"""Planning service — LangGraph-orchestrated agentic plan generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

from kuvox_ai.infrastructure import LLMClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.planning.models import (
    AddTextAction,
    DeleteItemsAction,
    MoveItemAction,
    PlanningRequest,
    SplitItemAction,
    TrimItemAction,
    UpdateAudioAction,
    UpdateSpeedAction,
    VideoEditorPlanningAction,
    VideoEditorPlanningRequest,
    VideoEditorPlanningResponse,
    VideoEditorTimelineItemSummary,
    VideoEditorTrackSummary,
)
from kuvox_ai.modules.retrieval import RetrievalService
from kuvox_ai.schemas import Plan

logger = get_logger(__name__)

_UNSUPPORTED_VIDEO_EDITOR_COMMAND = (
    "The AI planner can trim, split, delete, move, add text, set volume, or change speed."
)


@dataclass(frozen=True)
class _LocatedItem:
    item: VideoEditorTimelineItemSummary
    track: VideoEditorTrackSummary
    item_index: int


class PlanningService:
    """Public interface to the planning pipeline.

    Three paths are dispatched based on input classification:

    * **Direct path** — concrete commands → single LLM structured call → Plan.
    * **Reasoning path** — abstract commands → decompose → retrieve → synthesize.
    * **Code generation fallback** — out-of-schema requests → LLM emits Python
      executed in the sandbox.

    Every returned :class:`Plan` is validated against the operation schema
    before the call returns.
    """

    def __init__(self, *, llm: LLMClient, retrieval: RetrievalService) -> None:
        self._llm = llm
        self._retrieval = retrieval

    async def plan(self, request: PlanningRequest) -> Plan:
        """Produce a validated :class:`Plan` for the given request.

        TODO: implement LangGraph state machine selecting the appropriate
        path. All paths must converge on a Plan that round-trips through
        ``Plan.model_validate`` before being returned.
        """
        logger.info(
            "planning.plan.start",
            has_command=bool(request.command.strip()),
            context_shot_count=len(request.context_shot_ids),
        )
        raise NotImplementedError("PlanningService.plan is not implemented yet")

    async def plan_video_editor(
        self,
        request: VideoEditorPlanningRequest,
    ) -> VideoEditorPlanningResponse:
        """Produce deterministic action plans for concrete video-editor commands."""
        command = _normalize_spaces(request.command)
        logger.info(
            "planning.video_editor.start",
            has_command=bool(command),
            project_id=request.project_id,
            command_id=request.command_id,
        )
        if not command:
            return _unsupported(request, "Enter a video edit command.")

        lower = command.lower()
        if lower == "split here":
            playhead = request.playback.current_time if request.playback else request.playhead_time
            return _split_plan(request, "", playhead)

        if add_text := re.fullmatch(r'add\s+text\s+"([^"]+)"(?:\s+at\s+(.+))?', command, re.I):
            time = _parse_time(add_text.group(2)) if add_text.group(2) else (
                request.playback.current_time if request.playback else request.playhead_time
            )
            if time is None:
                return _unsupported(request, "Use a valid text start time, such as 4s.")
            return _success(
                request,
                actions=[AddTextAction(text=add_text.group(1), timeline_start=_round_time(time))],
                explanation=f"Added text at {_format_seconds(time)}.",
            )

        if trim_range := re.fullmatch(r"trim(?:\s+(.+?))?\s+from\s+(.+?)\s+to\s+(.+)", command, re.I):
            start = _parse_time(trim_range.group(2))
            end = _parse_time(trim_range.group(3))
            if start is None or end is None:
                return _unsupported(request, "Use a valid trim range, such as 5s to 12s.")
            return _trim_plan(request, trim_range.group(1) or "", start, end)

        if trim_duration := re.fullmatch(r"trim(?:\s+(.+?))?\s+to\s+(.+)", command, re.I):
            duration = _parse_time(trim_duration.group(2))
            if duration is None:
                return _unsupported(request, "Use a valid trim duration, such as 8s.")
            target = _resolve_single_target(request, trim_duration.group(1) or "", "any")
            if isinstance(target, str):
                return _unsupported(request, target)
            return _trim_plan(
                request,
                trim_duration.group(1) or "",
                target.item.timeline_start,
                target.item.timeline_start + duration,
            )

        if split := re.fullmatch(r"split(?:\s+(.+?))?\s+at\s+(.+)", command, re.I):
            time = _parse_time(split.group(2))
            if time is None:
                return _unsupported(request, "Use a valid split time, such as 12s.")
            return _split_plan(request, split.group(1) or "", time)

        if delete := re.fullmatch(r"delete(?:\s+(.+))?", command, re.I):
            return _delete_plan(request, delete.group(1) or "")

        if move := re.fullmatch(r"move(?:\s+(.+?))?\s+to\s+(.+)", command, re.I):
            time = _parse_time(move.group(2))
            if time is None:
                return _unsupported(request, "Use a valid move time, such as 18s.")
            return _move_plan(request, move.group(1) or "", time)

        if volume := re.fullmatch(r"set\s+volume(?:\s+(.+?))?\s+to\s+(\d+(?:\.\d+)?)\s*%", command, re.I):
            percent = float(volume.group(2))
            if percent < 0 or percent > 100:
                return _unsupported(request, "Volume must be between 0% and 100%.")
            target = _resolve_single_target(request, volume.group(1) or "", "audio")
            if isinstance(target, str):
                return _unsupported(request, target)
            return _success(
                request,
                actions=[UpdateAudioAction(item_id=target.item.id, volume=_round_time(percent / 100), muted=False)],
                explanation=f"Set {target.item.id} volume to {percent:g}%.",
            )

        if speed := re.fullmatch(r"change\s+speed(?:\s+(.+?))?\s+to\s+(\d+(?:\.\d+)?)\s*x", command, re.I):
            rate = float(speed.group(2))
            if rate <= 0:
                return _unsupported(request, "Speed must be greater than 0x.")
            target = _resolve_single_target(request, speed.group(1) or "", "clip")
            if isinstance(target, str):
                return _unsupported(request, target)
            return _success(
                request,
                actions=[UpdateSpeedAction(item_id=target.item.id, speed=_round_time(rate))],
                explanation=f"Changed {target.item.id} speed to {rate:g}x.",
            )

        return _unsupported(request, _UNSUPPORTED_VIDEO_EDITOR_COMMAND)


def _trim_plan(
    request: VideoEditorPlanningRequest,
    target_text: str,
    start: float,
    end: float,
) -> VideoEditorPlanningResponse:
    if end <= start:
        return _unsupported(request, "Trim end must be after trim start.")
    target = _resolve_single_target(request, target_text, "any")
    if isinstance(target, str):
        return _unsupported(request, target)

    item = target.item
    duration = _round_time(end - start)
    source_fields = _media_range_for_timeline_range(item, start, duration)
    if source_fields is None:
        return _unsupported(request, "That trim range falls outside the source media.")

    action = TrimItemAction(
        item_id=item.id,
        timeline_start=_round_time(start),
        duration=duration,
        source_in=source_fields[0],
        source_out=source_fields[1],
    )
    return _success(
        request,
        actions=[action],
        explanation=f"Trimmed {item.id} to {_format_seconds(start)}-{_format_seconds(end)}.",
    )


def _split_plan(
    request: VideoEditorPlanningRequest,
    target_text: str,
    split_time: float,
) -> VideoEditorPlanningResponse:
    target = _resolve_single_target(request, target_text, "any")
    if isinstance(target, str):
        return _unsupported(request, target)

    item = target.item
    offset = _round_time(split_time - item.timeline_start)
    if offset <= 0 or offset >= item.duration:
        return _unsupported(request, f"Split time must be inside {item.id}.")
    return _success(
        request,
        actions=[SplitItemAction(item_id=item.id, timeline_time=_round_time(split_time))],
        explanation=f"Split {item.id} at {_format_seconds(split_time)}.",
    )


def _delete_plan(
    request: VideoEditorPlanningRequest,
    target_text: str,
) -> VideoEditorPlanningResponse:
    if target_text.strip():
        target = _resolve_single_target(request, target_text, "any")
        if isinstance(target, str):
            return _unsupported(request, target)
        item_ids = [target.item.id]
    else:
        selected = _resolve_selected_targets(request)
        if isinstance(selected, str):
            return _unsupported(request, selected)
        item_ids = [located.item.id for located in selected]

    unique_item_ids = list(dict.fromkeys(item_ids))
    return _success(
        request,
        actions=[DeleteItemsAction(item_ids=unique_item_ids)],
        explanation=f"Deleted {len(unique_item_ids)} timeline item{'' if len(unique_item_ids) == 1 else 's'}.",
    )


def _move_plan(
    request: VideoEditorPlanningRequest,
    target_text: str,
    time: float,
) -> VideoEditorPlanningResponse:
    target = _resolve_single_target(request, target_text, "any")
    if isinstance(target, str):
        return _unsupported(request, target)
    return _success(
        request,
        actions=[MoveItemAction(item_id=target.item.id, timeline_start=_round_time(time))],
        explanation=f"Moved {target.item.id} to {_format_seconds(time)}.",
    )


def _resolve_single_target(
    request: VideoEditorPlanningRequest,
    target_text: str,
    target_kind: str,
) -> _LocatedItem | str:
    target = target_text.strip()
    matches = (
        _resolve_target_text(request, target, target_kind)
        if target
        else _resolve_default_target(request, target_kind)
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return "No matching timeline item was found."
    return "That target is ambiguous. Try an item id, clip 1, audio 1, or text 1."


def _resolve_selected_targets(request: VideoEditorPlanningRequest) -> list[_LocatedItem] | str:
    locations = {located.item.id: located for located in _all_located_items(request)}
    selected = [
        locations[item_id]
        for item_id in request.selection.selected_item_ids
        if item_id in locations
    ]
    if not selected:
        return "Select one or more timeline items to delete."
    return selected


def _resolve_default_target(
    request: VideoEditorPlanningRequest,
    target_kind: str,
) -> list[_LocatedItem]:
    locations = _all_located_items(request)
    if request.selection.active_item_id:
        active = next(
            (
                located
                for located in locations
                if located.item.id == request.selection.active_item_id
            ),
            None,
        )
        if active and _is_compatible_target(active.item, target_kind):
            return [active]

    selected = set(request.selection.selected_item_ids)
    return [
        located
        for located in locations
        if located.item.id in selected and _is_compatible_target(located.item, target_kind)
    ]


def _resolve_target_text(
    request: VideoEditorPlanningRequest,
    target_text: str,
    target_kind: str,
) -> list[_LocatedItem]:
    target = target_text.strip().lower()
    items = [
        located
        for located in _all_located_items(request)
        if _is_compatible_target(located.item, target_kind)
    ]
    exact = [located for located in items if located.item.id.lower() == target]
    if exact:
        return exact

    ordinal = re.fullmatch(r"(?:clip|video|audio|text|item)\s+(\d+)", target)
    if ordinal:
        index = int(ordinal.group(1)) - 1
        kind = target.split()[0]
        filtered = [
            located
            for located in items
            if kind in {"item"}
            or (kind in {"clip", "video"} and located.item.type == "video")
            or (kind == "audio" and located.item.type == "audio")
            or (kind == "text" and located.item.type == "text")
        ]
        return [filtered[index]] if 0 <= index < len(filtered) else []

    return [
        located
        for located in items
        if target in located.item.id.lower()
        or (located.item.text is not None and target in located.item.text.lower())
    ]


def _all_located_items(request: VideoEditorPlanningRequest) -> list[_LocatedItem]:
    located: list[_LocatedItem] = []
    for track in request.timeline.tracks:
        for index, item in enumerate(track.items):
            located.append(_LocatedItem(item=item, track=track, item_index=index))
    return located


def _is_compatible_target(item: VideoEditorTimelineItemSummary, target_kind: str) -> bool:
    if target_kind == "clip":
        return item.type == "video"
    if target_kind == "audio":
        return item.type == "audio"
    if target_kind == "text":
        return item.type == "text"
    return True


def _media_range_for_timeline_range(
    item: VideoEditorTimelineItemSummary,
    timeline_start: float,
    duration: float,
) -> tuple[float | None, float | None] | None:
    if item.type not in {"video", "audio"}:
        return (None, None)
    if item.source_in is None or item.source_out is None:
        return None
    speed = item.speed or 1.0
    source_in = _round_time(item.source_in + (timeline_start - item.timeline_start) * speed)
    source_out = _round_time(source_in + duration * speed)
    if source_in < item.source_in or source_out > item.source_out or source_out <= source_in:
        return None
    return (source_in, source_out)


def _success(
    request: VideoEditorPlanningRequest,
    *,
    actions: list[VideoEditorPlanningAction],
    explanation: str,
    warnings: list[str] | None = None,
) -> VideoEditorPlanningResponse:
    return VideoEditorPlanningResponse(
        ok=True,
        plan_id=f"plan-{uuid4()}",
        command_id=request.command_id,
        actions=actions,
        warnings=warnings or [],
        explanation=explanation,
        confidence=1.0,
    )


def _unsupported(request: VideoEditorPlanningRequest, reason: str) -> VideoEditorPlanningResponse:
    return VideoEditorPlanningResponse(
        ok=False,
        plan_id=f"plan-{uuid4()}",
        command_id=request.command_id,
        actions=[],
        warnings=[],
        explanation=reason,
        confidence=0.0,
        unsupported_reason=reason,
    )


def _parse_time(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?", normalized)
    if not match:
        return None
    return float(match.group(1))


def _round_time(value: float) -> float:
    return round(value, 3)


def _format_seconds(value: float) -> str:
    rounded = _round_time(value)
    return f"{rounded:g}s"


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
