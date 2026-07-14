"""Rendering service — executes Plans into finished videos."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from kuvox_ai.infrastructure import ObjectStorageClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.media_optimization import ffmpeg
from kuvox_ai.modules.rendering.models import RenderingMediaSource, RenderJob, RenderResult
from kuvox_ai.schemas import VideoRenderManifest
from kuvox_ai.schemas.render_manifest import (
    VideoRenderAnimationTrack,
    VideoRenderMediaSource,
    VideoRenderTextOverlay,
    VideoRenderVisualItem,
)

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

    def __init__(
        self, *, storage: ObjectStorageClient, work_dir: Path | str = Path("/tmp/kuvox-rendering")
    ) -> None:
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
            effect
            for effect in effects
            if isinstance(effect, dict) and effect.get("enabled") is not False
        ]
        if enabled_effects:
            raise RenderManifestError("Timeline effects are not supported by this renderer.")

    settings = _settings_for_manifest(job)
    document_media = document.get("media") if isinstance(document.get("media"), dict) else {}
    raw_tracks = document.get("tracks")
    tracks: list[Any] = raw_tracks if isinstance(raw_tracks, list) else []
    media_by_id = {source.media_id: source for source in job.media_sources}
    media_sources: dict[str, dict[str, Any]] = {}
    visual_items: list[dict[str, Any]] = []
    audio_items: list[dict[str, Any]] = []
    text_overlays: list[dict[str, Any]] = []
    stack_orders = _stack_orders(tracks)

    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict) or track.get("hidden") is True:
            continue
        track_kind = str(track.get("kind") or "")
        if track_kind == "audio" and track.get("muted") is True:
            continue
        raw_items = track.get("items")
        items: list[Any] = raw_items if isinstance(raw_items, list) else []
        track_id = str(track.get("id") or f"track-{track_index}")

        for item in items:
            if not isinstance(item, dict):
                continue
            advanced_error = _unsupported_advanced_state(item)
            if advanced_error:
                raise RenderManifestError(advanced_error)
            duration = _positive_float(item.get("duration"), "item.duration")
            if duration <= 0:
                continue
            item_type = str(item.get("type") or "")
            item_id = str(item.get("id") or "")

            if item_type == "text":
                text_overlays.append(
                    {
                        "itemId": item_id,
                        "trackId": track_id,
                        "text": str(item.get("text") or ""),
                        "timelineStart": _non_negative_float(
                            item.get("timelineStart"), "item.timelineStart"
                        ),
                        "duration": duration,
                        "style": _text_style(item.get("style")),
                        "transform": _transform(item.get("transform")),
                        "opacity": 1,
                        "layerOrder": _int_or(item.get("layerOrder"), track_index),
                        "stackOrder": stack_orders.get(item_id, 0),
                        **_animation_for_item(item),
                    }
                )
                continue

            if item_type not in {"video", "image", "overlay", "audio"}:
                continue
            media_id = str(item.get("mediaId") or "")
            source = media_by_id.get(media_id)
            if source is None:
                raise RenderManifestError(
                    f"Timeline item {item_id} references unresolved media {media_id}."
                )
            if (
                source.bucket_name is None
                or not source.bucket_name.strip()
                or not source.object_key.strip()
            ):
                raise RenderManifestError(f"Media source {media_id} is missing bucket/key.")

            media_ref = document_media.get(media_id) if isinstance(document_media, dict) else None
            if item_type != "audio" and (
                source.width is None
                or source.width <= 0
                or source.height is None
                or source.height <= 0
            ):
                raise RenderManifestError(
                    f"Media source {media_id} is missing dimensions required for rendering."
                )
            media_sources[media_id] = _manifest_media_source(media_id, media_ref, source)

            if item_type == "audio":
                if item.get("muted") is True:
                    continue
                audio_items.append(
                    {
                        "itemId": item_id,
                        "trackId": track_id,
                        "mediaId": media_id,
                        "timelineStart": _non_negative_float(
                            item.get("timelineStart"), "item.timelineStart"
                        ),
                        "duration": duration,
                        "sourceIn": _non_negative_float(item.get("sourceIn"), "item.sourceIn"),
                        "sourceOut": _positive_float(item.get("sourceOut"), "item.sourceOut"),
                        "speed": _positive_float(item.get("speed"), "item.speed", default=1),
                        "volume": _unit_float(item.get("volume"), default=1),
                        "muted": False,
                        "fades": _audio_fades(item.get("fades")),
                        "layerOrder": track_index,
                    }
                )
                continue

            visual = {
                "itemId": item_id,
                "trackId": track_id,
                "type": item_type,
                "mediaId": media_id,
                "timelineStart": _non_negative_float(
                    item.get("timelineStart"), "item.timelineStart"
                ),
                "duration": duration,
                "layerOrder": _int_or(item.get("layerOrder"), track_index),
                "stackOrder": stack_orders.get(item_id, 0),
                "transform": _transform(item.get("transform")),
                "crop": _crop(item.get("crop")),
                "opacity": _strict_unit_float(item.get("opacity"), "item.opacity", default=1),
                **_animation_for_item(item),
            }
            if item_type == "video":
                visual["sourceIn"] = _non_negative_float(item.get("sourceIn"), "item.sourceIn")
                visual["sourceOut"] = _positive_float(item.get("sourceOut"), "item.sourceOut")
                visual["speed"] = _positive_float(item.get("speed"), "item.speed")
                if isinstance(item.get("shotId"), str):
                    visual["shotId"] = item["shotId"]
            visual_items.append(visual)

    duration_seconds = max(
        [0.1]
        + [_item_end(item) for item in visual_items]
        + [_item_end(item) for item in audio_items]
        + [_item_end(item) for item in text_overlays]
    )

    manifest = VideoRenderManifest.model_validate(
        {
            "schemaVersion": 2,
            "projectId": str(document.get("projectId") or job.project_id),
            "settings": settings,
            "durationSeconds": round(duration_seconds, 3),
            "mediaSources": sorted(media_sources.values(), key=lambda item: str(item["mediaId"])),
            "visualItems": sorted(visual_items, key=_stack_sort_key),
            "audioItems": sorted(audio_items, key=_sort_key),
            "textOverlays": sorted(text_overlays, key=_stack_sort_key),
        }
    )
    _validate_manifest_animation_frames(manifest)

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
    started_at = time.perf_counter()
    silent_video = assets_dir / f"silent.{manifest.settings.format}"
    audio_path = assets_dir / ("audio.wav" if manifest.settings.format == "mov" else "audio.m4a")
    compose_started = time.perf_counter()
    timings = await asyncio.to_thread(
        _compose_and_encode_video, manifest, resolved_sources, silent_video
    )
    compose_finished = time.perf_counter()
    audio_started = time.perf_counter()
    has_audio = await _render_audio_track(manifest, resolved_sources, audio_path)
    audio_finished = time.perf_counter()
    if has_audio:
        await ffmpeg.run_command(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(silent_video),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-t",
                f"{max(0.1, manifest.duration_seconds):.6f}",
                str(output_path),
            ]
        )
    else:
        shutil.move(silent_video, output_path)
    finished_at = time.perf_counter()
    logger.info(
        "rendering.render.timing",
        frame_count=timings["frame_count"],
        decoding_seconds=round(timings["decoding_seconds"], 3),
        compositing_seconds=round(timings["compositing_seconds"], 3),
        encoding_seconds=round(timings["encoding_seconds"], 3),
        visual_seconds=round(compose_finished - compose_started, 3),
        audio_seconds=round(audio_finished - audio_started, 3),
        total_seconds=round(finished_at - started_at, 3),
    )


class _VideoFrameDecoder:
    def __init__(
        self, path: Path, width: int, height: int, source_in: float, fps: int, speed: float
    ) -> None:
        self._frame_bytes = width * height * 4
        self._process = subprocess.Popen(
            [
                ffmpeg.resolve_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{source_in:.9f}",
                "-i",
                str(path),
                "-an",
                "-vf",
                f"fps={fps / speed:.9f},scale={width}:{height}:flags=lanczos,format=rgba",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._width = width
        self._height = height

    def next_frame(self) -> Image.Image | None:
        if self._process.stdout is None:
            return None
        data = self._process.stdout.read(self._frame_bytes)
        if len(data) != self._frame_bytes:
            return None
        return Image.frombytes("RGBA", (self._width, self._height), data)

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.communicate()


def _compose_and_encode_video(
    manifest: VideoRenderManifest,
    resolved_sources: dict[str, _ResolvedSource],
    output_path: Path,
) -> dict[str, float]:
    width = manifest.settings.width
    height = manifest.settings.height
    fps = manifest.settings.frame_rate
    frame_count = max(1, math.ceil(max(0.1, manifest.duration_seconds) * fps))
    encoder = subprocess.Popen(
        [
            ffmpeg.resolve_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            *_video_codec_args(manifest),
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    decoders: dict[str, _VideoFrameDecoder] = {}
    image_cache: dict[str, Image.Image] = {}
    decoding_seconds = 0.0
    compositing_seconds = 0.0
    encode_started = time.perf_counter()
    stack: list[VideoRenderVisualItem | VideoRenderTextOverlay] = sorted(
        [*manifest.visual_items, *manifest.text_overlays],
        key=lambda item: (item.stack_order, item.item_id),
    )
    try:
        if encoder.stdin is None:
            raise ffmpeg.FfmpegError("FFmpeg raw-frame encoder did not expose stdin.")
        for frame_index in range(frame_count):
            timeline_time = frame_index / fps
            frame = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            composite_started = time.perf_counter()
            for item in stack:
                if not _active_at(item.timeline_start, item.duration, timeline_time):
                    continue
                item_time = max(0.0, timeline_time - item.timeline_start)
                if isinstance(item, VideoRenderVisualItem):
                    source = resolved_sources.get(item.media_id)
                    if source is None:
                        raise RenderManifestError(
                            f"Media source {item.media_id} was not downloaded."
                        )
                    decode_started = time.perf_counter()
                    if item.type == "video":
                        decoder = decoders.get(item.item_id)
                        if decoder is None:
                            media_source = _manifest_source(manifest, item.media_id)
                            decoder = _VideoFrameDecoder(
                                source.local_path,
                                media_source.width or 1,
                                media_source.height or 1,
                                (item.source_in or 0) + item_time * (item.speed or 1),
                                fps,
                                item.speed or 1,
                            )
                            decoders[item.item_id] = decoder
                        source_image = decoder.next_frame()
                    else:
                        source_image = image_cache.get(item.media_id)
                        if source_image is None:
                            source_image = Image.open(source.local_path).convert("RGBA")
                            media_source = _manifest_source(manifest, item.media_id)
                            expected_size = (media_source.width or 1, media_source.height or 1)
                            if source_image.size != expected_size:
                                source_image = source_image.resize(
                                    expected_size, Image.Resampling.LANCZOS
                                )
                            image_cache[item.media_id] = source_image
                    decoding_seconds += time.perf_counter() - decode_started
                    if source_image is not None:
                        layer, position = _render_visual_layer(
                            source_image, item, item_time, width, height
                        )
                        frame.alpha_composite(layer, position)
                else:
                    layer, position = _render_text_layer(item, item_time, manifest)
                    frame.alpha_composite(layer, position)
            compositing_seconds += time.perf_counter() - composite_started
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        _, stderr = encoder.communicate()
        if encoder.returncode != 0:
            raise ffmpeg.FfmpegError(stderr.decode("utf-8", errors="ignore").strip())
    finally:
        for decoder in decoders.values():
            decoder.close()
        if encoder.poll() is None:
            encoder.kill()
            encoder.communicate()
    return {
        "frame_count": float(frame_count),
        "decoding_seconds": decoding_seconds,
        "compositing_seconds": compositing_seconds,
        "encoding_seconds": max(
            0.0, time.perf_counter() - encode_started - compositing_seconds - decoding_seconds
        ),
    }


def _render_visual_layer(
    source: Image.Image,
    item: Any,
    item_time: float,
    project_width: int,
    project_height: int,
) -> tuple[Image.Image, tuple[int, int]]:
    transform = _evaluated_transform(item, item_time)
    crop = _evaluated_crop(item, item_time)
    opacity = _evaluated_opacity(item, item_time)
    source_width, source_height = source.size
    left = round(source_width * crop["left"])
    top = round(source_height * crop["top"])
    right = round(source_width * (1 - crop["right"]))
    bottom = round(source_height * (1 - crop["bottom"]))
    cropped = source.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
    base_scale = min(project_width / cropped.width, project_height / cropped.height)
    output_width = max(1, round(cropped.width * base_scale * abs(transform["scaleX"])))
    output_height = max(1, round(cropped.height * base_scale * abs(transform["scaleY"])))
    layer = cropped.resize((output_width, output_height), Image.Resampling.LANCZOS)
    if transform["scaleX"] < 0:
        layer = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform["scaleY"] < 0:
        layer = layer.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if opacity < 1:
        alpha = layer.getchannel("A").point(lambda value: round(value * opacity))
        layer.putalpha(alpha)
    if transform["rotation"]:
        layer = layer.rotate(-transform["rotation"], resample=Image.Resampling.BICUBIC, expand=True)
    center_x = project_width / 2 + transform["x"]
    center_y = project_height / 2 + transform["y"]
    return layer, (round(center_x - layer.width / 2), round(center_y - layer.height / 2))


def _render_text_layer(
    item: Any, item_time: float, manifest: VideoRenderManifest
) -> tuple[Image.Image, tuple[int, int]]:
    transform = _evaluated_transform(item, item_time)
    opacity = _evaluated_opacity(item, item_time)
    width = max(1, round(manifest.settings.width * 0.8 * transform["scaleX"]))
    height = max(1, round(manifest.settings.height * 0.12 * transform["scaleY"]))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if item.style.background_color:
        background = Image.new(
            "RGBA", layer.size, _rgba(item.style.background_color, round(255 * 0.72))
        )
        mask = Image.new("L", layer.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=6, fill=255)
        layer.alpha_composite(Image.composite(background, Image.new("RGBA", layer.size), mask))
    font_size = max(1, round(item.style.font_size * transform["scaleY"]))
    font = _font(font_size, item.style.font_family, item.style.font_weight, item.style.font_style)
    lines = _wrap_text(item.text, font, width)
    text = "\n".join(lines)
    draw = ImageDraw.Draw(layer)
    spacing = 4
    bbox = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, stroke_width=round(item.style.stroke_width or 0)
    )
    rendered_width = bbox[2] - bbox[0]
    rendered_height = bbox[3] - bbox[1]
    align = item.style.text_align or "center"
    x = (
        0
        if align == "left"
        else width - rendered_width
        if align == "right"
        else (width - rendered_width) / 2
    )
    y = (height - rendered_height) / 2 - bbox[1]
    shadow_blur = item.style.shadow_blur if item.style.shadow_blur is not None else 10
    shadow_color = item.style.shadow_color or "black"
    if shadow_blur > 0:
        shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).multiline_text(
            (x + (item.style.shadow_offset_x or 0), y + (item.style.shadow_offset_y or 0)),
            text,
            font=font,
            fill=_rgba(shadow_color, round(255 * 0.55)),
            align=align,
            spacing=spacing,
        )
        layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(shadow_blur)))
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=_rgba(item.style.color, 255),
        align=align,
        spacing=spacing,
        stroke_width=round(item.style.stroke_width or 0),
        stroke_fill=_rgba(item.style.stroke_color or item.style.color, 255),
    )
    if opacity < 1:
        layer.putalpha(layer.getchannel("A").point(lambda value: round(value * opacity)))
    origin = (
        round(manifest.settings.width * 0.1 + transform["x"]),
        round(manifest.settings.height * 0.5 + transform["y"]),
    )
    if transform["rotation"]:
        rotated = layer.rotate(
            -transform["rotation"], resample=Image.Resampling.BICUBIC, expand=True
        )
        radians = math.radians(transform["rotation"])
        corners = [(0, 0), (width, 0), (0, height), (width, height)]
        min_x = min(x * math.cos(radians) - y * math.sin(radians) for x, y in corners)
        min_y = min(x * math.sin(radians) + y * math.cos(radians) for x, y in corners)
        return rotated, (round(origin[0] + min_x), round(origin[1] + min_y))
    return layer, origin


async def _render_audio_track(
    manifest: VideoRenderManifest,
    resolved_sources: dict[str, _ResolvedSource],
    output_path: Path,
) -> bool:
    if not manifest.audio_items:
        return False
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    input_indices: dict[str, int] = {}
    for item in manifest.audio_items:
        if item.media_id in input_indices:
            continue
        resolved = resolved_sources.get(item.media_id)
        if resolved is None:
            raise RenderManifestError(f"Media source {item.media_id} was not downloaded.")
        input_indices[item.media_id] = len(input_indices)
        args.extend(["-i", str(resolved.local_path)])
    filters: list[str] = []
    labels: list[str] = []
    duration = max(0.1, manifest.duration_seconds)
    for index, item in enumerate(manifest.audio_items):
        label = f"a{index}"
        item_duration = min(
            item.duration, max(0.001, (item.source_out - item.source_in) / item.speed)
        )
        parts = [
            f"[{input_indices[item.media_id]}:a]atrim=start={item.source_in:.6f}:end={item.source_out:.6f}",
            "asetpts=PTS-STARTPTS",
            *(_atempo_filters(item.speed) if item.speed != 1 else []),
            f"atrim=duration={item_duration:.6f}",
            f"volume={item.volume:.6f}",
        ]
        if item.fades.fade_in_duration > 0:
            parts.append(f"afade=t=in:st=0:d={item.fades.fade_in_duration:.6f}")
        if item.fades.fade_out_duration > 0:
            parts.append(
                f"afade=t=out:st={max(0, item_duration - item.fades.fade_out_duration):.6f}:d={item.fades.fade_out_duration:.6f}"
            )
        delay_ms = max(0, round(item.timeline_start * 1000))
        parts.append(f"adelay={delay_ms}|{delay_ms}")
        filters.append(",".join(parts) + f"[{label}]")
        labels.append(label)
    joined = "".join(f"[{label}]" for label in labels)
    mix = (
        f"{joined}amix=inputs={len(labels)}:duration=longest:dropout_transition=0"
        if len(labels) > 1
        else f"[{labels[0]}]anull"
    )
    filters.append(f"{mix},apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}[aout]")
    args.extend(["-filter_complex", ";".join(filters), "-map", "[aout]"])
    if manifest.settings.format == "mov":
        args.extend(["-c:a", "pcm_s16le"])
    else:
        args.extend(["-c:a", "aac", "-b:a", "192k"])
    args.append(str(output_path))
    await ffmpeg.run_command(args)
    return True


def _active_at(timeline_start: float, duration: float, timeline_time: float) -> bool:
    return timeline_start <= timeline_time < timeline_start + duration


def _validate_manifest_animation_frames(manifest: VideoRenderManifest) -> None:
    fps = manifest.settings.frame_rate
    sources = {source.media_id: source for source in manifest.media_sources}
    for item in manifest.visual_items:
        source = sources[item.media_id]
        width = source.width or 0
        height = source.height or 0
        for frame_index in range(max(1, math.ceil(item.duration * fps)) + 1):
            item_time = min(item.duration, frame_index / fps)
            transform = _evaluated_transform(item, item_time)
            crop = _evaluated_crop(item, item_time)
            opacity = _evaluated_opacity(item, item_time)
            invalid = (
                transform["scaleX"] <= 0
                or transform["scaleY"] <= 0
                or not 0 <= opacity <= 1
                or any(not 0 <= value <= 1 for value in crop.values())
                or (1 - crop["left"] - crop["right"]) * width < 1
                or (1 - crop["top"] - crop["bottom"]) * height < 1
            )
            if invalid:
                raise RenderManifestError(
                    f"Animated state on {item.item_id} is invalid at output frame {frame_index}."
                )
    for text_item in manifest.text_overlays:
        for frame_index in range(max(1, math.ceil(text_item.duration * fps)) + 1):
            item_time = min(text_item.duration, frame_index / fps)
            transform = _evaluated_transform(text_item, item_time)
            opacity = _evaluated_opacity(text_item, item_time)
            if transform["scaleX"] > 0 and transform["scaleY"] > 0 and 0 <= opacity <= 1:
                continue
            raise RenderManifestError(
                f"Animated state on {text_item.item_id} is invalid at output frame {frame_index}."
            )


def _manifest_source(manifest: VideoRenderManifest, media_id: str) -> VideoRenderMediaSource:
    for source in manifest.media_sources:
        if source.media_id == media_id:
            return source
    raise RenderManifestError(f"Manifest media source {media_id} is missing.")


def evaluate_animation_track(
    track: VideoRenderAnimationTrack | None,
    item_time: float,
    fallback: float,
) -> float:
    if track is None or not track.keyframes:
        return fallback
    keyframes = sorted(track.keyframes, key=lambda keyframe: keyframe.time)
    if item_time <= keyframes[0].time:
        return float(keyframes[0].value)
    if item_time >= keyframes[-1].time:
        return float(keyframes[-1].value)
    right_index = next(
        index for index, keyframe in enumerate(keyframes) if keyframe.time >= item_time
    )
    left = keyframes[right_index - 1]
    right = keyframes[right_index]
    progress = (item_time - left.time) / max(
        float.fromhex("0x1.0000000000000p-52"), right.time - left.time
    )
    easing = right.easing
    if easing is not None:
        inverse = 1 - progress
        progress = (
            3 * inverse * inverse * progress * easing[1]
            + 3 * inverse * progress * progress * easing[3]
            + progress * progress * progress
        )
    return float(left.value + (right.value - left.value) * progress)


def _evaluated_transform(item: Any, item_time: float) -> dict[str, float]:
    animation = item.animation.transform if item.animation and item.animation.transform else None
    return {
        "x": evaluate_animation_track(
            animation.x if animation else None, item_time, item.transform.x
        ),
        "y": evaluate_animation_track(
            animation.y if animation else None, item_time, item.transform.y
        ),
        "scaleX": evaluate_animation_track(
            animation.scale_x if animation else None, item_time, item.transform.scale_x
        ),
        "scaleY": evaluate_animation_track(
            animation.scale_y if animation else None, item_time, item.transform.scale_y
        ),
        "rotation": evaluate_animation_track(
            animation.rotation if animation else None, item_time, item.transform.rotation
        ),
    }


def _evaluated_crop(item: Any, item_time: float) -> dict[str, float]:
    animation = item.animation.crop if item.animation and item.animation.crop else None
    return {
        "top": evaluate_animation_track(
            animation.top if animation else None, item_time, item.crop.top
        ),
        "right": evaluate_animation_track(
            animation.right if animation else None, item_time, item.crop.right
        ),
        "bottom": evaluate_animation_track(
            animation.bottom if animation else None, item_time, item.crop.bottom
        ),
        "left": evaluate_animation_track(
            animation.left if animation else None, item_time, item.crop.left
        ),
    }


def _evaluated_opacity(item: Any, item_time: float) -> float:
    track = item.animation.opacity if item.animation else None
    return evaluate_animation_track(track, item_time, item.opacity)


def _wrap_text(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _video_codec_args(manifest: VideoRenderManifest) -> list[str]:
    if manifest.settings.preset == "prores-master" and manifest.settings.format == "mov":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    crf = {"draft": "28", "standard": "23", "high": "18"}[manifest.settings.quality]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def _settings_for_manifest(job: RenderJob) -> dict[str, Any]:
    settings = dict(job.settings)
    document_settings = (
        job.document_json.get("settings") if isinstance(job.document_json, dict) else {}
    )
    if not isinstance(document_settings, dict):
        document_settings = {}
    width = int(settings.get("width") or document_settings.get("width") or 1920)
    height = int(settings.get("height") or document_settings.get("height") or 1080)
    frame_rate = int(settings.get("frameRate") or document_settings.get("frameRate") or 30)
    return {
        "preset": str(
            settings.get("preset") or document_settings.get("exportPreset") or "h264-1080p"
        ),
        "format": str(settings.get("format") or "mp4"),
        "resolution": str(settings.get("resolution") or _resolution_for(width, height)),
        "width": width,
        "height": height,
        "frameRate": frame_rate,
        "quality": str(settings.get("quality") or "standard"),
        "destinationLabel": str(settings.get("destinationLabel") or "Kuvox render"),
    }


def _manifest_media_source(
    media_id: str, media_ref: Any, source: RenderingMediaSource
) -> dict[str, Any]:
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


def _font(
    size: int,
    family: str = "Inter",
    weight: str | None = None,
    style: str | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    bold = weight in {"semibold", "bold"}
    italic = style == "italic"
    candidates = _font_candidates(family, bold=bold, italic=italic)
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, size=size)
            if hasattr(font, "set_variation_by_name"):
                variation = {
                    "medium": "Medium",
                    "semibold": "SemiBold",
                    "bold": "Bold",
                }.get(weight or "", "Regular")
                with suppress(OSError, ValueError):
                    font.set_variation_by_name(variation)
            return font
        except OSError:
            continue
    return ImageFont.load_default()


def _font_candidates(family: str, *, bold: bool, italic: bool) -> tuple[str, ...]:
    catalog = _font_catalog()
    requested = family.strip() or "Inter"
    entry = catalog.get(requested) or catalog["Inter"]
    if "fallback" in entry:
        entry = catalog.get(entry["fallback"], catalog["Inter"])
    key = (
        "italic"
        if italic and "italic" in entry
        else "bold"
        if bold and "bold" in entry
        else "regular"
    )
    primary = _font_assets_dir() / entry[key]
    fallback = _font_assets_dir() / catalog["Inter"]["regular"]
    return (str(primary), str(fallback), "DejaVuSans.ttf")


def _font_assets_dir() -> Path:
    return Path(__file__).with_name("fonts")


@lru_cache(maxsize=1)
def _font_catalog() -> dict[str, dict[str, str]]:
    with (_font_assets_dir() / "catalog.json").open(encoding="utf-8") as file:
        return cast(dict[str, dict[str, str]], json.load(file))


def _rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError:
        rgb = (255, 255, 255)
    if len(rgb) == 4:
        return (rgb[0], rgb[1], rgb[2], min(alpha, rgb[3]))
    return (rgb[0], rgb[1], rgb[2], alpha)


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
        "backgroundColor": style.get("backgroundColor")
        if isinstance(style.get("backgroundColor"), str)
        else None,
        "fontWeight": style.get("fontWeight")
        if style.get("fontWeight") in {"normal", "medium", "semibold", "bold"}
        else "normal",
        "fontStyle": style.get("fontStyle")
        if style.get("fontStyle") in {"normal", "italic"}
        else "normal",
        "textAlign": style.get("textAlign")
        if style.get("textAlign") in {"left", "center", "right"}
        else "center",
        "strokeColor": style.get("strokeColor")
        if isinstance(style.get("strokeColor"), str)
        else None,
        "strokeWidth": _non_negative_float(
            style.get("strokeWidth"), "style.strokeWidth", default=0
        ),
        "shadowColor": style.get("shadowColor")
        if isinstance(style.get("shadowColor"), str)
        else None,
        "shadowBlur": _non_negative_float(style.get("shadowBlur"), "style.shadowBlur", default=10),
        "shadowOffsetX": _float_or(style.get("shadowOffsetX"), 0),
        "shadowOffsetY": _float_or(style.get("shadowOffsetY"), 0),
        "animType": style.get("animType") if isinstance(style.get("animType"), str) else None,
        "animDur": _non_negative_float(style.get("animDur"), "style.animDur", default=0),
    }


def _stack_orders(tracks: list[Any]) -> dict[str, int]:
    candidates: list[tuple[int, int, int, int, str]] = []
    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict) or track.get("hidden") is True:
            continue
        track_kind = str(track.get("kind") or "")
        raw_items = track.get("items")
        items: list[Any] = raw_items if isinstance(raw_items, list) else []
        for item_index, item in enumerate(items):
            if not isinstance(item, dict) or str(item.get("type") or "") == "audio":
                continue
            item_type = str(item.get("type") or "")
            phase = 0 if track_kind == "video" and item_type in {"video", "image"} else 1
            layer_order = _int_or(item.get("layerOrder"), 0)
            primary_order = -track_index if phase == 0 else layer_order
            secondary_order = layer_order if phase == 0 else track_index
            candidates.append(
                (
                    phase,
                    primary_order,
                    secondary_order,
                    item_index,
                    str(item.get("id") or ""),
                )
            )
    candidates.sort()
    return {candidate[4]: index for index, candidate in enumerate(candidates)}


def _unsupported_advanced_state(item: dict[str, Any]) -> str | None:
    advanced = item.get("advanced")
    if not isinstance(advanced, dict) or not advanced:
        return None
    item_id = str(item.get("id") or "<unknown>")
    blockers = (
        ("trackingTargets", "Tracking metadata"),
        ("autoReframe", "Auto-reframe metadata"),
        ("color", "Color processing"),
        ("freezeFrames", "Freeze frames"),
        ("timeRemap", "Time remapping"),
    )
    for key, label in blockers:
        if advanced.get(key):
            return f"{label} on {item_id} is not supported by export."
    raw_transform = advanced.get("transform")
    transform: dict[str, Any] = raw_transform if isinstance(raw_transform, dict) else {}
    if transform.get("anchorX") or transform.get("anchorY"):
        return f"Anchor animation on {item_id} is not supported by export."
    if item.get("type") == "text" and advanced.get("crop"):
        return f"Crop animation on text item {item_id} is not supported by export."
    tracks = [*transform.values()]
    raw_crop = advanced.get("crop")
    crop: dict[str, Any] = raw_crop if isinstance(raw_crop, dict) else {}
    if item.get("type") != "text":
        tracks.extend(crop.values())
    if advanced.get("opacity") is not None:
        tracks.append(advanced["opacity"])
    if not tracks or any(
        not isinstance(track, dict) or not track.get("keyframes") for track in tracks
    ):
        return (
            f"Advanced state on {item_id} must contain supported transform, crop, "
            "or opacity keyframes."
        )
    return None


def _animation_for_item(item: dict[str, Any]) -> dict[str, Any]:
    advanced = item.get("advanced")
    if not isinstance(advanced, dict):
        return {}
    transform = _animation_properties(
        advanced.get("transform"), ("x", "y", "scaleX", "scaleY", "rotation")
    )
    crop = (
        None
        if item.get("type") == "text"
        else _animation_properties(advanced.get("crop"), ("top", "right", "bottom", "left"))
    )
    opacity = _animation_track(advanced.get("opacity"))
    animation = {
        key: value
        for key, value in {
            "transform": transform,
            "crop": crop,
            "opacity": opacity,
        }.items()
        if value is not None
    }
    return {"animation": animation} if animation else {}


def _animation_properties(value: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
    properties = value if isinstance(value, dict) else {}
    normalized = {
        key: track for key in keys if (track := _animation_track(properties.get(key))) is not None
    }
    return normalized or None


def _animation_track(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("keyframes"), list):
        return None
    keyframes: list[dict[str, Any]] = []
    for keyframe in cast(list[Any], value["keyframes"]):
        if not isinstance(keyframe, dict):
            continue
        normalized: dict[str, Any] = {
            "time": _non_negative_float(keyframe.get("time"), "keyframe.time"),
            "value": _float_or(keyframe.get("value"), 0) or 0,
        }
        easing = keyframe.get("easing")
        if isinstance(easing, list) and len(easing) == 4:
            normalized["easing"] = [float(entry) for entry in easing]
        keyframes.append(normalized)
    keyframes.sort(key=lambda keyframe: float(keyframe["time"]))
    return {"keyframes": keyframes} if keyframes else None


def _audio_fades(value: Any) -> dict[str, float]:
    fades = value if isinstance(value, dict) else {}
    return {
        "fadeInDuration": _non_negative_float(
            fades.get("fadeInDuration"), "fades.fadeInDuration", default=0
        ),
        "fadeOutDuration": _non_negative_float(
            fades.get("fadeOutDuration"), "fades.fadeOutDuration", default=0
        ),
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
        "top": _strict_unit_float(crop.get("top"), "crop.top", default=0),
        "right": _strict_unit_float(crop.get("right"), "crop.right", default=0),
        "bottom": _strict_unit_float(crop.get("bottom"), "crop.bottom", default=0),
        "left": _strict_unit_float(crop.get("left"), "crop.left", default=0),
    }


def _item_end(item: dict[str, Any]) -> float:
    return float(item["timelineStart"]) + float(item["duration"])


def _sort_key(item: dict[str, Any]) -> tuple[float, int, str]:
    return (float(item["timelineStart"]), int(item["layerOrder"]), str(item["itemId"]))


def _stack_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    return (int(item["stackOrder"]), str(item["itemId"]))


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
    return (
        "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:128]
        or "job"
    )


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


def _strict_unit_float(value: Any, name: str, *, default: float) -> float:
    parsed = _float_or(value, default)
    if parsed is None or not 0 <= parsed <= 1:
        raise RenderManifestError(f"{name} must be between 0 and 1.")
    return parsed


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
