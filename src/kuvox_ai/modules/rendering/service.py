"""Rendering service — executes Plans into finished videos."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont

from kuvox_ai.infrastructure import ObjectStorageClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.media_optimization import ffmpeg
from kuvox_ai.modules.rendering.models import RenderJob, RenderingMediaSource, RenderResult
from kuvox_ai.schemas import VideoRenderManifest

logger = get_logger(__name__)


class RenderManifestError(ValueError):
    """Raised when a saved editor document cannot be rendered by this worker."""


@dataclass(frozen=True)
class _ResolvedSource:
    source: RenderingMediaSource
    local_path: Path


class RenderingService:
    """Public interface to the rendering pipeline.

    Walks the :class:`Plan`'s operations, drives MoviePy / FFmpeg to produce
    the output file, and uploads it to object storage.
    """

    def __init__(self, *, storage: ObjectStorageClient, work_dir: Path | str = Path("/tmp/kuvox-rendering")) -> None:
        self._storage = storage
        self._work_dir = Path(work_dir)

    async def render(self, job: RenderJob) -> RenderResult:
        """Execute one render job end-to-end."""
        logger.info(
            "rendering.render.start",
            render_job_id=job.render_job_id,
            n_media_sources=len(job.media_sources),
        )

        job_dir = self._work_dir / _safe_path_part(job.render_job_id)
        sources_dir = job_dir / "sources"
        assets_dir = job_dir / "assets"
        output_path = job_dir / f"output.{_output_extension(job)}"
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            sources_dir.mkdir(parents=True, exist_ok=False)
            assets_dir.mkdir(parents=True, exist_ok=False)

            manifest = build_manifest(job)
            resolved_sources = await self._download_sources(job, sources_dir)
            await _render_manifest(
                manifest,
                resolved_sources,
                assets_dir=assets_dir,
                output_path=output_path,
            )
            await self._storage.upload_file(
                output_path,
                job.output_bucket_name,
                job.output_storage_key,
                content_type=job.output_content_type,
            )
            size_bytes = output_path.stat().st_size
            logger.info(
                "rendering.render.completed",
                render_job_id=job.render_job_id,
                output_bucket_name=job.output_bucket_name,
                output_storage_key=job.output_storage_key,
                output_size_bytes=size_bytes,
            )
            return RenderResult(
                render_job_id=job.render_job_id,
                output_bucket_name=job.output_bucket_name,
                output_storage_key=job.output_storage_key,
                output_content_type=job.output_content_type,
                output_size_bytes=size_bytes,
                duration_seconds=manifest.duration_seconds,
            )
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    async def _download_sources(
        self,
        job: RenderJob,
        sources_dir: Path,
    ) -> dict[str, _ResolvedSource]:
        resolved: dict[str, _ResolvedSource] = {}
        for source in job.media_sources:
            if source.bucket_name is None or not source.bucket_name.strip():
                raise RenderManifestError(f"Media source {source.media_id} is missing bucketName.")
            if not source.object_key.strip():
                raise RenderManifestError(f"Media source {source.media_id} is missing objectKey.")

            extension = Path(source.object_key).suffix or _extension_for_kind(source.kind)
            local_path = sources_dir / f"{_safe_path_part(source.media_id)}{extension}"
            await self._storage.download_file(source.bucket_name, source.object_key, local_path)
            resolved[source.media_id] = _ResolvedSource(source=source, local_path=local_path)
        return resolved


def build_manifest(job: RenderJob) -> VideoRenderManifest:
    """Build and validate the renderer manifest from the API event payload."""

    document = job.document_json
    if not isinstance(document, dict):
        raise RenderManifestError("documentJson must be an object.")

    transitions = document.get("transitions")
    if isinstance(transitions, list) and transitions:
        raise RenderManifestError("Timeline transitions are not supported by this renderer.")

    effects = document.get("effects")
    if isinstance(effects, list):
        enabled_effects = [
            effect for effect in effects
            if isinstance(effect, dict) and effect.get("enabled") is not False
        ]
        if enabled_effects:
            raise RenderManifestError("Timeline effects are not supported by this renderer.")

    settings = _settings_for_manifest(job)
    document_media = document.get("media") if isinstance(document.get("media"), dict) else {}
    tracks = document.get("tracks") if isinstance(document.get("tracks"), list) else []
    media_by_id = {source.media_id: source for source in job.media_sources}
    media_sources: dict[str, dict[str, Any]] = {}
    visual_items: list[dict[str, Any]] = []
    audio_items: list[dict[str, Any]] = []
    text_overlays: list[dict[str, Any]] = []

    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict) or track.get("hidden") is True:
            continue
        track_kind = str(track.get("kind") or "")
        if track_kind == "audio" and track.get("muted") is True:
            continue
        items = track.get("items") if isinstance(track.get("items"), list) else []
        track_id = str(track.get("id") or f"track-{track_index}")

        for item in items:
            if not isinstance(item, dict):
                continue
            duration = _positive_float(item.get("duration"), "item.duration")
            if duration <= 0:
                continue
            item_type = str(item.get("type") or "")
            item_id = str(item.get("id") or "")

            if item_type == "text":
                text_overlays.append({
                    "itemId": item_id,
                    "trackId": track_id,
                    "text": str(item.get("text") or ""),
                    "timelineStart": _non_negative_float(item.get("timelineStart"), "item.timelineStart"),
                    "duration": duration,
                    "style": _text_style(item.get("style")),
                    "transform": _transform(item.get("transform")),
                    "opacity": 1,
                    "layerOrder": _int_or(item.get("layerOrder"), track_index),
                })
                continue

            if item_type not in {"video", "image", "overlay", "audio"}:
                continue
            media_id = str(item.get("mediaId") or "")
            source = media_by_id.get(media_id)
            if source is None:
                raise RenderManifestError(f"Timeline item {item_id} references unresolved media {media_id}.")
            if source.bucket_name is None or not source.bucket_name.strip() or not source.object_key.strip():
                raise RenderManifestError(f"Media source {media_id} is missing bucket/key.")

            media_ref = document_media.get(media_id) if isinstance(document_media, dict) else None
            media_sources[media_id] = _manifest_media_source(media_id, media_ref, source)

            if item_type == "audio":
                if item.get("muted") is True:
                    continue
                audio_items.append({
                    "itemId": item_id,
                    "trackId": track_id,
                    "mediaId": media_id,
                    "timelineStart": _non_negative_float(item.get("timelineStart"), "item.timelineStart"),
                    "duration": duration,
                    "sourceIn": _non_negative_float(item.get("sourceIn"), "item.sourceIn"),
                    "sourceOut": _positive_float(item.get("sourceOut"), "item.sourceOut"),
                    "speed": _positive_float(item.get("speed"), "item.speed", default=1),
                    "volume": _unit_float(item.get("volume"), default=1),
                    "muted": False,
                    "fades": _audio_fades(item.get("fades")),
                    "layerOrder": track_index,
                })
                continue

            visual = {
                "itemId": item_id,
                "trackId": track_id,
                "type": item_type,
                "mediaId": media_id,
                "timelineStart": _non_negative_float(item.get("timelineStart"), "item.timelineStart"),
                "duration": duration,
                "layerOrder": _int_or(item.get("layerOrder"), track_index),
                "transform": _transform(item.get("transform")),
                "opacity": _unit_float(item.get("opacity"), default=1),
            }
            if item_type == "video":
                visual["sourceIn"] = _non_negative_float(item.get("sourceIn"), "item.sourceIn")
                visual["sourceOut"] = _positive_float(item.get("sourceOut"), "item.sourceOut")
                visual["speed"] = _positive_float(item.get("speed"), "item.speed")
                visual["crop"] = _crop(item.get("crop"))
                if isinstance(item.get("shotId"), str):
                    visual["shotId"] = item["shotId"]
            visual_items.append(visual)

    duration_seconds = max(
        [0.1]
        + [_item_end(item) for item in visual_items]
        + [_item_end(item) for item in audio_items]
        + [_item_end(item) for item in text_overlays]
    )

    manifest = VideoRenderManifest.model_validate({
        "schemaVersion": 1,
        "projectId": str(document.get("projectId") or job.project_id),
        "settings": settings,
        "durationSeconds": round(duration_seconds, 3),
        "mediaSources": sorted(media_sources.values(), key=lambda item: str(item["mediaId"])),
        "visualItems": sorted(visual_items, key=_sort_key),
        "audioItems": sorted(audio_items, key=_sort_key),
        "textOverlays": sorted(text_overlays, key=_sort_key),
    })

    if not manifest.visual_items and not manifest.text_overlays:
        raise RenderManifestError("Timeline has no visible visual or text items to render.")
    return manifest


async def _render_manifest(
    manifest: VideoRenderManifest,
    resolved_sources: dict[str, _ResolvedSource],
    *,
    assets_dir: Path,
    output_path: Path,
) -> None:
    width = manifest.settings.width
    height = manifest.settings.height
    fps = manifest.settings.frame_rate
    duration = max(0.1, manifest.duration_seconds)
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration:.3f}",
    ]
    input_indices: dict[str, int] = {}
    next_input_index = 1

    used_media_ids = {item.media_id for item in manifest.visual_items} | {item.media_id for item in manifest.audio_items}
    for media_id in sorted(used_media_ids):
        resolved = resolved_sources.get(media_id)
        if resolved is None:
            raise RenderManifestError(f"Media source {media_id} was not downloaded.")
        input_indices[media_id] = next_input_index
        next_input_index += 1
        if _manifest_kind(resolved.source.kind) == "image":
            args.extend(["-loop", "1", "-t", f"{duration:.3f}", "-i", str(resolved.local_path)])
        else:
            args.extend(["-i", str(resolved.local_path)])

    text_indices: dict[str, int] = {}
    for index, overlay in enumerate(manifest.text_overlays):
        asset_path = assets_dir / f"text-{index}.png"
        _write_text_overlay(asset_path, overlay, manifest)
        text_indices[overlay.item_id] = next_input_index
        next_input_index += 1
        args.extend(["-loop", "1", "-t", f"{duration:.3f}", "-i", str(asset_path)])

    filter_parts: list[str] = []
    current_video = "0:v"
    doc_width = max(1, int(manifest.settings.width))
    doc_height = max(1, int(manifest.settings.height))
    visual_specs: list[tuple[float, int, str, str, str]] = []

    for item in manifest.visual_items:
        source = resolved_sources[item.media_id].source
        input_index = input_indices[item.media_id]
        source_width = max(1, source.width or manifest.settings.width)
        source_height = max(1, source.height or manifest.settings.height)
        crop = item.crop
        crop_left = crop.left if crop else 0
        crop_right = crop.right if crop else 0
        crop_top = crop.top if crop else 0
        crop_bottom = crop.bottom if crop else 0
        cropped_width = max(1, round(source_width * (1 - crop_left - crop_right)))
        cropped_height = max(1, round(source_height * (1 - crop_top - crop_bottom)))
        base_scale = min(width / cropped_width, height / cropped_height)
        output_width = max(1, round(cropped_width * base_scale * item.transform.scale_x))
        output_height = max(1, round(cropped_height * base_scale * item.transform.scale_y))
        x = round((width - output_width) / 2 + item.transform.x * (width / doc_width))
        y = round((height - output_height) / 2 + item.transform.y * (height / doc_height))
        label = f"visual{len(visual_specs)}"
        filters = [f"[{input_index}:v]"]
        if item.type == "video":
            source_in = item.source_in or 0
            source_out = item.source_out or (source_in + item.duration)
            speed = item.speed or 1
            filters.append(f"trim=start={source_in:.6f}:end={source_out:.6f},setpts=(PTS-STARTPTS)/{speed:.6f}")
        else:
            filters.append(f"trim=duration={item.duration:.6f},setpts=PTS-STARTPTS")
        if crop:
            filters.append(
                f"crop=w=iw*{1 - crop_left - crop_right:.6f}:h=ih*{1 - crop_top - crop_bottom:.6f}:"
                f"x=iw*{crop_left:.6f}:y=ih*{crop_top:.6f}"
            )
        filters.append(f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease")
        filters.append("format=rgba")
        if item.opacity < 1:
            filters.append(f"colorchannelmixer=aa={item.opacity:.6f}")
        if item.transform.rotation:
            filters.append(f"rotate={math.radians(item.transform.rotation):.8f}:c=none:ow=rotw(iw):oh=roth(ih)")
        filters.append(f"setpts=PTS+{item.timeline_start:.6f}/TB")
        filter_parts.append(",".join(filters) + f"[{label}]")
        visual_specs.append((item.timeline_start, item.layer_order, label, str(x), str(y)))

    for overlay in manifest.text_overlays:
        input_index = text_indices[overlay.item_id]
        label = f"text{len(visual_specs)}"
        text_width, text_height = _text_overlay_size(overlay, manifest)
        x = round((width - text_width) / 2 + overlay.transform.x * (width / doc_width))
        y = round((height - text_height) / 2 + overlay.transform.y * (height / doc_height))
        filters = [f"[{input_index}:v]", f"trim=duration={overlay.duration:.6f}", "setpts=PTS-STARTPTS", "format=rgba"]
        if overlay.opacity < 1:
            filters.append(f"colorchannelmixer=aa={overlay.opacity:.6f}")
        if overlay.transform.rotation:
            filters.append(f"rotate={math.radians(overlay.transform.rotation):.8f}:c=none:ow=rotw(iw):oh=roth(ih)")
        filters.append(f"setpts=PTS+{overlay.timeline_start:.6f}/TB")
        filter_parts.append(",".join(filters) + f"[{label}]")
        visual_specs.append((overlay.timeline_start, overlay.layer_order, label, str(x), str(y)))

    for _, _, label, x, y in sorted(visual_specs):
        next_label = f"v{len(filter_parts)}"
        filter_parts.append(f"[{current_video}][{label}]overlay=x={x}:y={y}:eof_action=pass:shortest=0[{next_label}]")
        current_video = next_label

    audio_labels: list[str] = []
    for index, item in enumerate(manifest.audio_items):
        input_index = input_indices[item.media_id]
        label = f"a{index}"
        item_duration = min(item.duration, max(0.001, (item.source_out - item.source_in) / item.speed))
        fade_out_start = max(0, item_duration - item.fades.fade_out_duration)
        filters = [
            f"[{input_index}:a]",
            f"atrim=start={item.source_in:.6f}:end={item.source_out:.6f}",
            "asetpts=PTS-STARTPTS",
        ]
        if item.speed != 1:
            filters.extend(_atempo_filters(item.speed))
        filters.append(f"atrim=duration={item_duration:.6f}")
        filters.append(f"volume={item.volume:.6f}")
        if item.fades.fade_in_duration > 0:
            filters.append(f"afade=t=in:st=0:d={item.fades.fade_in_duration:.6f}")
        if item.fades.fade_out_duration > 0:
            filters.append(f"afade=t=out:st={fade_out_start:.6f}:d={item.fades.fade_out_duration:.6f}")
        delay_ms = max(0, round(item.timeline_start * 1000))
        filters.append(f"adelay={delay_ms}|{delay_ms}")
        filter_parts.append(",".join(filters) + f"[{label}]")
        audio_labels.append(label)

    map_args = ["-map", f"[{current_video}]"]
    if audio_labels:
        if len(audio_labels) == 1:
            filter_parts.append(f"[{audio_labels[0]}]apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}[aout]")
        else:
            joined = "".join(f"[{label}]" for label in audio_labels)
            filter_parts.append(
                f"{joined}amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=0,"
                f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}[aout]"
            )
        map_args.extend(["-map", "[aout]"])

    filter_complex = ";".join(filter_parts)
    if filter_complex:
        args.extend(["-filter_complex", filter_complex])
    args.extend(map_args)
    args.extend(["-t", f"{duration:.3f}", "-r", str(fps)])
    args.extend(_codec_args(manifest))
    args.append(str(output_path))
    await ffmpeg.run_command(args)


def _settings_for_manifest(job: RenderJob) -> dict[str, Any]:
    settings = dict(job.settings)
    document_settings = job.document_json.get("settings") if isinstance(job.document_json, dict) else {}
    if not isinstance(document_settings, dict):
        document_settings = {}
    width = int(settings.get("width") or document_settings.get("width") or 1920)
    height = int(settings.get("height") or document_settings.get("height") or 1080)
    frame_rate = int(settings.get("frameRate") or document_settings.get("frameRate") or 30)
    return {
        "preset": str(settings.get("preset") or document_settings.get("exportPreset") or "h264-1080p"),
        "format": str(settings.get("format") or "mp4"),
        "resolution": str(settings.get("resolution") or _resolution_for(width, height)),
        "width": width,
        "height": height,
        "frameRate": frame_rate,
        "quality": str(settings.get("quality") or "standard"),
        "destinationLabel": str(settings.get("destinationLabel") or "Kuvox render"),
    }


def _manifest_media_source(media_id: str, media_ref: Any, source: RenderingMediaSource) -> dict[str, Any]:
    ref = media_ref if isinstance(media_ref, dict) else {}
    return {
        "mediaId": media_id,
        "kind": _manifest_kind(source.kind),
        "name": str(ref.get("name") or Path(source.object_key).name or media_id),
        "durationSeconds": source.duration_seconds,
        "width": source.width,
        "height": source.height,
        "mimeType": source.content_type,
        "canonical": {"variant": "canonical", "url": "", "storageKey": source.object_key},
    }


def _write_text_overlay(path: Path, overlay: Any, manifest: VideoRenderManifest) -> None:
    text_width, text_height = _text_overlay_size(overlay, manifest)
    image = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_size = max(1, round(overlay.style.font_size * overlay.transform.scale_y))
    font = _font(font_size)
    if overlay.style.background_color:
        draw.rounded_rectangle(
            [(0, 0), (text_width - 1, text_height - 1)],
            radius=8,
            fill=_rgba(overlay.style.background_color, 210),
        )
    bbox = draw.multiline_textbbox((0, 0), overlay.text, font=font, spacing=4)
    rendered_width = bbox[2] - bbox[0]
    rendered_height = bbox[3] - bbox[1]
    align = overlay.style.text_align or "center"
    if align == "left":
        x = 16
    elif align == "right":
        x = max(16, text_width - rendered_width - 16)
    else:
        x = max(0, (text_width - rendered_width) / 2)
    y = max(0, (text_height - rendered_height) / 2 - bbox[1])
    draw.multiline_text((x, y), overlay.text, font=font, fill=_rgba(overlay.style.color, 255), align=align, spacing=4)
    image.save(path)


def _text_overlay_size(overlay: Any, manifest: VideoRenderManifest) -> tuple[int, int]:
    font_size = max(1, round(overlay.style.font_size * overlay.transform.scale_y))
    width = max(32, round(manifest.settings.width * 0.75 * overlay.transform.scale_x))
    height = max(font_size + 24, round(font_size * 1.8))
    return width, height


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError:
        rgb = (255, 255, 255)
    if len(rgb) == 4:
        return (rgb[0], rgb[1], rgb[2], min(alpha, rgb[3]))
    return (rgb[0], rgb[1], rgb[2], alpha)


def _codec_args(manifest: VideoRenderManifest) -> list[str]:
    if manifest.settings.preset == "prores-master" and manifest.settings.format == "mov":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le"]
    crf = {"draft": "28", "standard": "23", "high": "18"}[manifest.settings.quality]
    return [
        "-c:v", "libx264", "-preset", "medium", "-crf", crf, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
    ]


def _atempo_filters(speed: float) -> list[str]:
    values: list[float] = []
    remaining = speed
    while remaining > 2:
        values.append(2)
        remaining /= 2
    while remaining < 0.5:
        values.append(0.5)
        remaining /= 0.5
    values.append(remaining)
    return [f"atempo={value:.6f}" for value in values]


def _text_style(value: Any) -> dict[str, Any]:
    style = value if isinstance(value, dict) else {}
    return {
        "fontFamily": str(style.get("fontFamily") or "Inter"),
        "fontSize": _positive_float(style.get("fontSize"), "style.fontSize", default=48),
        "color": str(style.get("color") or "#ffffff"),
        "backgroundColor": style.get("backgroundColor") if isinstance(style.get("backgroundColor"), str) else None,
        "fontWeight": style.get("fontWeight") if style.get("fontWeight") in {"normal", "medium", "semibold", "bold"} else "normal",
        "fontStyle": style.get("fontStyle") if style.get("fontStyle") in {"normal", "italic"} else "normal",
        "textAlign": style.get("textAlign") if style.get("textAlign") in {"left", "center", "right"} else "center",
    }


def _audio_fades(value: Any) -> dict[str, float]:
    fades = value if isinstance(value, dict) else {}
    return {
        "fadeInDuration": _non_negative_float(fades.get("fadeInDuration"), "fades.fadeInDuration", default=0),
        "fadeOutDuration": _non_negative_float(fades.get("fadeOutDuration"), "fades.fadeOutDuration", default=0),
    }


def _transform(value: Any) -> dict[str, float]:
    transform = value if isinstance(value, dict) else {}
    return {
        "x": _float_or(transform.get("x"), 0) or 0,
        "y": _float_or(transform.get("y"), 0) or 0,
        "scaleX": _positive_float(transform.get("scaleX"), "transform.scaleX", default=1),
        "scaleY": _positive_float(transform.get("scaleY"), "transform.scaleY", default=1),
        "rotation": _float_or(transform.get("rotation"), 0) or 0,
    }


def _crop(value: Any) -> dict[str, float]:
    crop = value if isinstance(value, dict) else {}
    return {
        "top": _unit_float(crop.get("top"), default=0),
        "right": _unit_float(crop.get("right"), default=0),
        "bottom": _unit_float(crop.get("bottom"), default=0),
        "left": _unit_float(crop.get("left"), default=0),
    }


def _item_end(item: dict[str, Any]) -> float:
    return float(item["timelineStart"]) + float(item["duration"])


def _sort_key(item: dict[str, Any]) -> tuple[float, int, str]:
    return (float(item["timelineStart"]), int(item["layerOrder"]), str(item["itemId"]))


def _resolution_for(width: int, height: int) -> str:
    if (width, height) == (1280, 720):
        return "1280x720"
    if (width, height) == (1920, 1080):
        return "1920x1080"
    if (width, height) == (3840, 2160):
        return "3840x2160"
    return "current"


def _manifest_kind(value: str) -> str:
    normalized = value.lower()
    if normalized in {"video", "audio", "image"}:
        return normalized
    if normalized.endswith(".video") or normalized == "0":
        return "video"
    if normalized.endswith(".audio") or normalized == "1":
        return "audio"
    if normalized.endswith(".image") or normalized == "2":
        return "image"
    return normalized


def _extension_for_kind(kind: str) -> str:
    normalized = _manifest_kind(kind)
    if normalized == "image":
        return ".png"
    if normalized == "audio":
        return ".wav"
    return ".mp4"


def _output_extension(job: RenderJob) -> str:
    fmt = str(job.settings.get("format") or job.output_format or "mp4").lower()
    return "mov" if fmt == "mov" else "mp4"


def _safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:128] or "job"


def _positive_float(value: Any, name: str, default: float | None = None) -> float:
    parsed = _float_or(value, default)
    if parsed is None or parsed <= 0:
        raise RenderManifestError(f"{name} must be positive.")
    return parsed


def _non_negative_float(value: Any, name: str, default: float | None = None) -> float:
    parsed = _float_or(value, default)
    if parsed is None or parsed < 0:
        raise RenderManifestError(f"{name} must be non-negative.")
    return parsed


def _unit_float(value: Any, *, default: float) -> float:
    parsed = _float_or(value, default)
    if parsed is None:
        return default
    return min(1, max(0, parsed))


def _float_or(value: Any, default: float | None) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
