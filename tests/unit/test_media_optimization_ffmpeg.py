from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kuvox_ai.modules.media_optimization import ffmpeg


def test_resolve_ffprobe_uses_sibling_binary_when_path_is_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    ffmpeg_exe = binary_dir / "ffmpeg.exe"
    ffprobe_exe = binary_dir / "ffprobe.exe"
    ffmpeg_exe.write_bytes(b"")
    ffprobe_exe.write_bytes(b"")

    monkeypatch.setattr(
        "kuvox_ai.modules.media_optimization.ffmpeg.shutil.which", lambda _name: None
    )
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: str(ffmpeg_exe)),
    )

    assert ffmpeg.resolve_ffprobe_exe() == str(ffprobe_exe)


async def test_ffprobe_json_uses_resolved_ffprobe_binary(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(ffmpeg, "resolve_ffprobe_exe", lambda: "resolved-ffprobe")
    monkeypatch.setattr("kuvox_ai.modules.media_optimization.ffmpeg.subprocess.run", fake_run)

    assert await ffmpeg.ffprobe_json(tmp_path / "video.mp4") == {}
    assert calls[0][0] == "resolved-ffprobe"
