from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from kuvox_ai.modules.media_optimization import ffmpeg
from kuvox_ai.modules.rendering import RenderingService
from kuvox_ai.modules.rendering.models import RenderJob
from kuvox_ai.modules.rendering.service import (
    RenderManifestError,
    _render_visual_layer,
    _VideoFrameDecoder,
    build_manifest,
    evaluate_animation_track,
)
from kuvox_ai.schemas.render_manifest import VideoRenderAnimationTrack


async def test_render_image_only_uploads_playable_mp4(
    mock_storage: AsyncMock, tmp_path: Path
) -> None:
    try:
        ffmpeg.resolve_ffmpeg_exe()
    except ffmpeg.FfmpegError as exc:
        pytest.skip(f"FFmpeg unavailable in this environment: {exc}")

    source_path = tmp_path / "source.png"
    Image.new("RGB", (320, 180), color=(32, 96, 160)).save(source_path)

    async def download_file(bucket: str, key: str, destination: Path) -> None:
        assert bucket == "kuvox-canonical"
        assert key == "media/image/canonical.png"
        shutil.copyfile(source_path, destination)

    mock_storage.download_file.side_effect = download_file
    svc = RenderingService(storage=mock_storage, work_dir=tmp_path / "render-work")
    job = render_job()

    result = await svc.render(job)

    assert result.output_bucket_name == "kuvox-renders"
    assert result.output_storage_key == "renders/job-1.mp4"
    assert result.output_content_type == "video/mp4"
    assert result.output_size_bytes > 0
    mock_storage.upload_file.assert_awaited_once()
    upload_args = mock_storage.upload_file.await_args.args
    assert Path(upload_args[0]).name == "output.mp4"
    assert upload_args[1:] == ("kuvox-renders", "renders/job-1.mp4")
    assert mock_storage.upload_file.await_args.kwargs == {"content_type": "video/mp4"}


def test_build_manifest_rejects_unsupported_transitions() -> None:
    job = render_job()
    job.document_json["transitions"] = [{"id": "t1", "type": "crossfade"}]

    with pytest.raises(RenderManifestError, match="transitions"):
        build_manifest(job)


def test_build_manifest_rejects_missing_source_bucket() -> None:
    job = render_job()
    job.media_sources[0].bucket_name = None

    with pytest.raises(RenderManifestError, match="bucket"):
        build_manifest(job)


def test_animation_evaluator_matches_typescript_hold_and_easing() -> None:
    track = VideoRenderAnimationTrack.model_validate(
        {
            "keyframes": [
                {"time": 0, "value": 0},
                {"time": 2, "value": 100, "easing": [0.42, 0, 0.58, 1]},
            ]
        }
    )
    assert evaluate_animation_track(track, -1, 50) == 0
    assert evaluate_animation_track(track, 1, 50) == pytest.approx(50)
    assert evaluate_animation_track(track, 2, 50) == 100
    assert evaluate_animation_track(track, 3, 50) == 100


def test_visual_layer_preserves_center_with_independent_scale_crop_and_rotation() -> None:
    job = render_job()
    item = build_manifest(job).visual_items[0]
    item.transform.scale_x = 0.5
    item.transform.scale_y = 1.25
    item.transform.rotation = 30
    item.crop.left = 0.25
    source = Image.new("RGBA", (320, 180), (255, 0, 0, 128))

    layer, position = _render_visual_layer(source, item, 0, 320, 180)

    assert layer.mode == "RGBA"
    assert position[0] + layer.width / 2 == pytest.approx(160, abs=1)
    assert position[1] + layer.height / 2 == pytest.approx(90, abs=1)
    assert layer.getchannel("A").getextrema()[1] <= 128


def test_video_decoder_samples_source_time_at_output_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_popen(args: list[str], **_kwargs: object) -> SimpleNamespace:
        captured.extend(args)
        return SimpleNamespace(
            stdout=BytesIO(b""),
            stderr=BytesIO(b""),
            poll=lambda: 0,
            terminate=lambda: None,
            communicate=lambda **_kwargs: (b"", b""),
            kill=lambda: None,
        )

    monkeypatch.setattr(
        "kuvox_ai.modules.rendering.service.ffmpeg.resolve_ffmpeg_exe", lambda: "ffmpeg"
    )
    monkeypatch.setattr("kuvox_ai.modules.rendering.service.subprocess.Popen", fake_popen)

    decoder = _VideoFrameDecoder(Path("source.mp4"), 320, 180, 3.5, 24, 2)
    decoder.close()

    assert captured[captured.index("-ss") + 1] == "3.500000000"
    assert "fps=12.000000000" in captured[captured.index("-vf") + 1]


def render_job() -> RenderJob:
    return RenderJob.model_validate(
        {
            "eventId": "evt-1",
            "eventType": "rendering.requested",
            "occurredAt": "2026-07-09T00:00:00Z",
            "renderJobId": "job-1",
            "timelineId": "timeline-1",
            "projectId": "project-1",
            "revisionId": "revision-1",
            "revisionNumber": 1,
            "requestedByUserId": "user-1",
            "settings": {
                "preset": "h264-720p",
                "format": "mp4",
                "resolution": "1280x720",
                "width": 320,
                "height": 180,
                "frameRate": 24,
                "quality": "draft",
                "destinationLabel": "Test render",
            },
            "documentJson": {
                "schemaVersion": 1,
                "projectId": "project-1",
                "settings": {
                    "width": 320,
                    "height": 180,
                    "frameRate": 24,
                    "exportPreset": "h264-720p",
                },
                "media": {
                    "image-1": {
                        "id": "image-1",
                        "kind": "image",
                        "name": "Still.png",
                        "width": 320,
                        "height": 180,
                    }
                },
                "tracks": [
                    {
                        "id": "v1",
                        "kind": "video",
                        "label": "V1",
                        "hidden": False,
                        "muted": False,
                        "items": [
                            {
                                "id": "still-1",
                                "type": "image",
                                "mediaId": "image-1",
                                "timelineStart": 0,
                                "duration": 0.5,
                                "transform": {
                                    "x": 0,
                                    "y": 0,
                                    "scaleX": 1,
                                    "scaleY": 1,
                                    "rotation": 0,
                                },
                                "opacity": 1,
                                "layerOrder": 0,
                            }
                        ],
                    }
                ],
                "transitions": [],
                "effects": [],
            },
            "mediaSources": [
                {
                    "mediaId": "image-1",
                    "kind": "image",
                    "bucketName": "kuvox-canonical",
                    "objectKey": "media/image/canonical.png",
                    "contentType": "image/png",
                    "sizeBytes": 123,
                    "durationSeconds": None,
                    "width": 320,
                    "height": 180,
                }
            ],
            "outputBucketName": "kuvox-renders",
            "outputStorageKey": "renders/job-1.mp4",
            "outputContentType": "video/mp4",
        }
    )
