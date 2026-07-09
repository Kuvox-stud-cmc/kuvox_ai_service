"""Small async wrappers for FFmpeg and FFprobe."""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast


class FfmpegError(RuntimeError):
    """Raised when an FFmpeg/FFprobe command fails."""


class _ImageioFfmpegModule(Protocol):
    def get_ffmpeg_exe(self) -> str: ...


@lru_cache(maxsize=1)
def resolve_ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path and _ffmpeg_candidate_works(path):
        return path

    imageio_path = _resolve_imageio_ffmpeg_exe()
    if imageio_path:
        return imageio_path

    if path:
        raise FfmpegError(f"FFmpeg executable on PATH is not usable: {path}")

    raise FfmpegError("FFmpeg executable not found on PATH.")


def _resolve_imageio_ffmpeg_exe() -> str | None:
    try:
        imageio_ffmpeg = cast(
            _ImageioFfmpegModule,
            importlib.import_module("imageio_ffmpeg"),
        )
    except ImportError:
        return None

    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_candidate_works(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0


def resolve_ffprobe_exe() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path

    ffmpeg_path = Path(resolve_ffmpeg_exe())
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
    ffprobe_path = ffmpeg_path.with_name(ffprobe_name)
    if ffprobe_path.exists():
        return str(ffprobe_path)

    return "ffprobe"


async def run_command(args: list[str], timeout_seconds: int = 900) -> None:
    def _run() -> None:
        resolved_args = args.copy()
        if resolved_args and resolved_args[0] == "ffmpeg":
            resolved_args[0] = resolve_ffmpeg_exe()

        try:
            result = subprocess.run(
                resolved_args,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise FfmpegError("FFmpeg command timed out.") from exc
        except OSError as exc:
            raise FfmpegError(str(exc)) from exc

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            stdout = result.stdout.decode("utf-8", errors="ignore").strip()
            details = stderr or stdout or "FFmpeg produced no stderr/stdout."
            command = " ".join(str(part) for part in resolved_args)
            raise FfmpegError(
                f"FFmpeg command failed with exit code {result.returncode}: {command}\n{details}"
            )

    await asyncio.to_thread(_run)


async def ffprobe_json(path: Path) -> dict[str, Any]:
    def _run() -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                [
                    resolve_ffprobe_exe(),
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise FfmpegError(str(exc)) from exc

    result = await asyncio.to_thread(_run)

    if result.returncode != 0:
        raise FfmpegError(result.stderr.decode("utf-8", errors="ignore"))

    parsed = json.loads(result.stdout.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise FfmpegError("FFprobe returned an unexpected payload.")
    return parsed
