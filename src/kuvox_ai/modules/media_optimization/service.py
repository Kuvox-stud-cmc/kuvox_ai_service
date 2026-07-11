"""Media storage optimization service."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from kuvox_ai.infrastructure.object_storage_client import ObjectStorageClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.media_optimization import ffmpeg
from kuvox_ai.modules.media_optimization.models import (
    MediaKind,
    MediaOptimizationCompleted,
    MediaOptimizationRequested,
    OptimizedObject,
)

logger = get_logger(__name__)


class MediaOptimizationService:
    """Download raw media, optimize it with FFmpeg, and upload deterministic outputs."""

    def __init__(
        self,
        *,
        storage: ObjectStorageClient,
        canonical_bucket: str,
        proxy_bucket: str,
        thumbnail_bucket: str,
        work_dir: Path,
        video_canonical_crf: int = 28,
        video_proxy_crf: int = 30,
        video_proxy_max_width: int = 1280,
        image_max_width: int = 1920,
        thumbnail_width: int = 320,
    ) -> None:
        self._storage = storage
        self._canonical_bucket = canonical_bucket
        self._proxy_bucket = proxy_bucket
        self._thumbnail_bucket = thumbnail_bucket
        self._work_dir = work_dir
        self._video_canonical_crf = video_canonical_crf
        self._video_proxy_crf = video_proxy_crf
        self._video_proxy_max_width = video_proxy_max_width
        self._image_max_width = image_max_width
        self._thumbnail_width = thumbnail_width

    async def optimize(self, request: MediaOptimizationRequested) -> MediaOptimizationCompleted:
        self._work_dir.mkdir(parents=True, exist_ok=True)

        with temporary_job_dir(self._work_dir) as job_dir:
            input_path = job_dir / safe_filename(request.original_file_name)

            await self._storage.download_file(request.bucket_name, request.object_key, input_path)

            if request.kind == MediaKind.video:
                return await self._optimize_video(request, input_path, job_dir)
            if request.kind == MediaKind.audio:
                return await self._optimize_audio(request, input_path, job_dir)
            if request.kind == MediaKind.image:
                return await self._optimize_image(request, input_path, job_dir)

            raise ValueError(f"Unsupported media kind: {request.kind}")

    async def _optimize_video(
        self,
        request: MediaOptimizationRequested,
        input_path: Path,
        job_dir: Path,
    ) -> MediaOptimizationCompleted:
        canonical_path = job_dir / "canonical.mp4"
        proxy_path = job_dir / "proxy.mp4"
        poster_path = job_dir / "poster.png"

        await ffmpeg.run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-crf",
                str(self._video_canonical_crf),
                "-preset",
                "medium",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(canonical_path),
            ]
        )
        await run_optional_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                f"scale='min({self._video_proxy_max_width},iw)':-2",
                "-c:v",
                "libx264",
                "-crf",
                str(self._video_proxy_crf),
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(proxy_path),
            ]
        )
        await run_optional_command(
            [
                "ffmpeg",
                "-y",
                "-ss",
                "00:00:00.1",
                "-i",
                str(input_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
                "-c:v",
                "png",
                str(poster_path),
            ]
        )

        base_key = output_base_key(request)
        canonical = await self._upload_optimized(
            canonical_path,
            self._canonical_bucket,
            f"{base_key}/canonical.mp4",
            "video/mp4",
        )
        proxy = await self._upload_optional_optimized(
            proxy_path,
            self._proxy_bucket,
            f"{base_key}/proxy.mp4",
            "video/mp4",
        )
        thumbnail = await self._upload_optional_optimized(
            poster_path,
            self._thumbnail_bucket,
            f"{base_key}/poster.png",
            "image/png",
        )
        metadata = await extract_basic_metadata_safely(canonical_path)

        return self._completed(
            request,
            canonical=canonical,
            proxy=proxy,
            thumbnail=thumbnail,
            metadata=metadata,
        )

    async def _optimize_audio(
        self,
        request: MediaOptimizationRequested,
        input_path: Path,
        job_dir: Path,
    ) -> MediaOptimizationCompleted:
        canonical_path = job_dir / "canonical.opus"
        waveform_path = job_dir / "waveform.png"
        await ffmpeg.run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-map_metadata",
                "-1",
                "-c:a",
                "libopus",
                "-b:a",
                "64k",
                str(canonical_path),
            ]
        )
        await run_optional_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-filter_complex",
                "showwavespic=s=640x240:colors=0x4f46e5",
                "-frames:v",
                "1",
                "-c:v",
                "png",
                str(waveform_path),
            ]
        )

        base_key = output_base_key(request)
        canonical = await self._upload_optimized(
            canonical_path,
            self._canonical_bucket,
            f"{base_key}/canonical.opus",
            "audio/opus",
        )
        thumbnail = await self._upload_optional_optimized(
            waveform_path,
            self._thumbnail_bucket,
            f"{base_key}/waveform.png",
            "image/png",
        )
        metadata = await extract_basic_metadata_safely(canonical_path)
        return self._completed(
            request,
            canonical=canonical,
            thumbnail=thumbnail,
            metadata=metadata,
        )

    async def _optimize_image(
        self,
        request: MediaOptimizationRequested,
        input_path: Path,
        job_dir: Path,
    ) -> MediaOptimizationCompleted:
        canonical_path = job_dir / "canonical.webp"
        thumb_path = job_dir / "thumb.webp"

        await write_webp_image_variant(
            input_path,
            canonical_path,
            max_width=self._image_max_width,
            quality=80,
        )
        await write_webp_image_variant(
            input_path,
            thumb_path,
            max_width=self._thumbnail_width,
            quality=70,
        )

        base_key = output_base_key(request)
        canonical = await self._upload_optimized(
            canonical_path,
            self._canonical_bucket,
            f"{base_key}/canonical.webp",
            "image/webp",
        )
        thumbnail = await self._upload_optimized(
            thumb_path,
            self._thumbnail_bucket,
            f"{base_key}/thumb.webp",
            "image/webp",
        )
        metadata = image_metadata(canonical_path)
        return self._completed(
            request,
            canonical=canonical,
            thumbnail=thumbnail,
            metadata=metadata,
        )

    async def _upload_optimized(
        self,
        source: Path,
        bucket: str,
        key: str,
        content_type: str,
    ) -> OptimizedObject:
        await self._storage.upload_file(source, bucket, key, content_type=content_type)
        stat = await asyncio.to_thread(source.stat)
        return OptimizedObject(
            bucket_name=bucket,
            object_key=key,
            content_type=content_type,
            size_bytes=stat.st_size,
        )

    async def _upload_optional_optimized(
        self,
        source: Path,
        bucket: str,
        key: str,
        content_type: str,
    ) -> OptimizedObject | None:
        if not await asyncio.to_thread(source.exists):
            return None

        try:
            return await self._upload_optimized(source, bucket, key, content_type=content_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "media_optimization.optional_upload_failed",
                bucket=bucket,
                key=key,
                error=str(exc),
            )
            return None

    def _completed(
        self,
        request: MediaOptimizationRequested,
        *,
        canonical: OptimizedObject | None = None,
        proxy: OptimizedObject | None = None,
        thumbnail: OptimizedObject | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaOptimizationCompleted:
        data = metadata or {}
        return MediaOptimizationCompleted(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            canonical=canonical,
            proxy=proxy,
            thumbnail=thumbnail,
            duration_seconds=data.get("durationSeconds"),
            width=data.get("width"),
            height=data.get("height"),
            frame_rate=data.get("frameRate"),
            codec=data.get("codec"),
            raw_bucket_name=request.bucket_name,
            raw_object_key=request.object_key,
            raw_size_bytes=request.size_bytes,
        )


def safe_filename(filename: str) -> str:
    return Path(filename).name or "input"


@contextmanager
def temporary_job_dir(work_dir: Path) -> Iterator[Path]:
    job_dir = work_dir / f"job-{uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield job_dir
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def output_base_key(request: MediaOptimizationRequested) -> str:
    return f"media/{request.media_id}"


async def run_optional_command(args: list[str]) -> bool:
    try:
        await ffmpeg.run_command(args)
        return True
    except ffmpeg.FfmpegError as exc:
        logger.warning(
            "media_optimization.optional_ffmpeg_failed",
            output=args[-1] if args else None,
            error=str(exc),
        )
        return False


async def extract_basic_metadata_safely(path: Path) -> dict[str, Any]:
    try:
        return extract_basic_metadata(await ffmpeg.ffprobe_json(path))
    except ffmpeg.FfmpegError as exc:
        logger.warning(
            "media_optimization.ffprobe_failed",
            path=str(path),
            error=str(exc),
        )
        return {}


def extract_basic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    streams = metadata.get("streams", [])
    if not isinstance(streams, list):
        streams = []

    video_stream = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"),
        None,
    )
    format_data = metadata.get("format", {})
    duration = format_data.get("duration") if isinstance(format_data, dict) else None

    if video_stream is not None:
        return {
            "durationSeconds": _optional_float(duration),
            "width": _optional_int(video_stream.get("width")),
            "height": _optional_int(video_stream.get("height")),
            "frameRate": _parse_frame_rate(video_stream.get("avg_frame_rate")),
            "codec": _optional_str(video_stream.get("codec_name")),
        }

    if audio_stream is not None:
        return {
            "durationSeconds": _optional_float(duration),
            "codec": _optional_str(audio_stream.get("codec_name")),
        }

    return {}


def image_metadata(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {}

    with Image.open(path) as image:
        width, height = image.size

    return {
        "width": width,
        "height": height,
    }


async def write_webp_image_variant(
    source: Path,
    destination: Path,
    *,
    max_width: int,
    quality: int,
) -> None:
    await asyncio.to_thread(
        _write_webp_image_variant_sync,
        source,
        destination,
        max_width,
        quality,
    )


def _write_webp_image_variant_sync(
    source: Path,
    destination: Path,
    max_width: int,
    quality: int,
) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image optimization.") from exc

    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image)
        if normalized.mode not in {"RGB", "RGBA"}:
            normalized = normalized.convert("RGBA" if _has_alpha(normalized) else "RGB")

        width, height = normalized.size
        if width > max_width:
            target_height = max(1, round(height * (max_width / width)))
            normalized = normalized.resize((max_width, target_height), Image.Resampling.LANCZOS)

        destination.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(destination, format="WEBP", quality=quality, method=6)


def _has_alpha(image: Any) -> bool:
    return image.mode in {"LA", "RGBA"} or (
        image.mode == "P" and "transparency" in getattr(image, "info", {})
    )


def _parse_frame_rate(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    try:
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
