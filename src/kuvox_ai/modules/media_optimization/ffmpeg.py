"""Small async wrappers for FFmpeg and FFprobe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class FfmpegError(RuntimeError):
    """Raised when an FFmpeg/FFprobe command fails."""


async def run_command(args: list[str], timeout_seconds: int = 900) -> None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise FfmpegError("FFmpeg command timed out.") from exc

    if process.returncode != 0:
        raise FfmpegError(stderr.decode("utf-8", errors="ignore"))


async def ffprobe_json(path: Path) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise FfmpegError(stderr.decode("utf-8", errors="ignore"))

    parsed = json.loads(stdout.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise FfmpegError("FFprobe returned an unexpected payload.")
    return parsed
