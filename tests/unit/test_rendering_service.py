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
    _evaluated_opacity,
    _render_audio_track,
    _render_visual_layer,
    _ResolvedSource,
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


@pytest.mark.parametrize(
    ("output_format", "content_type", "audio_codec"),
    [("mp4", "video/mp4", "aac"), ("mov", "video/quicktime", "pcm_s16le")],
)
async def test_render_video_preserves_embedded_audio_and_output_metadata(
    mock_storage: AsyncMock,
    tmp_path: Path,
    output_format: str,
    content_type: str,
    audio_codec: str,
) -> None:
    try:
        ffmpeg.resolve_ffmpeg_exe()
    except ffmpeg.FfmpegError as exc:
        pytest.skip(f"FFmpeg unavailable in this environment: {exc}")

    source_path = tmp_path / "source-with-audio.mp4"
    await ffmpeg.run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=#205080:s=160x90:r=24:d=0.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.6",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source_path),
        ]
    )
    captured_output = tmp_path / f"captured.{output_format}"
    expected_content_type = content_type

    async def download_file(_bucket: str, _key: str, destination: Path) -> None:
        shutil.copyfile(source_path, destination)

    async def upload_file(source: Path, _bucket: str, _key: str, *, content_type: str) -> None:
        assert content_type == expected_content_type
        shutil.copyfile(source, captured_output)

    mock_storage.download_file.side_effect = download_file
    mock_storage.upload_file.side_effect = upload_file
    job = video_render_job(output_format, content_type)
    svc = RenderingService(storage=mock_storage, work_dir=tmp_path / f"render-{output_format}")

    result = await svc.render(job)
    probe = await ffmpeg.ffprobe_json(captured_output)

    assert result.output_content_type == content_type
    streams = probe["streams"]
    video_stream = next(stream for stream in streams if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert (video_stream["width"], video_stream["height"]) == (160, 90)
    assert video_stream["codec_name"] == "h264"
    assert video_stream["r_frame_rate"] == "24/1"
    assert audio_stream["codec_name"] == audio_codec
    assert float(probe["format"]["duration"]) == pytest.approx(0.5, abs=0.08)


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


def test_manifest_v3_carries_logical_canvas_style_and_visual_fades() -> None:
    job = render_job()
    item = job.document_json["tracks"][0]["items"][0]
    item["properties"] = {
        "adjust": {"exposure": {"value": 12}, "temperature": {"value": 8}},
        "filters": {
            "filterType": {"value": "Cinematic"},
            "intensity": {"value": 50},
            "blend": {"value": 80},
        },
        "animation": {"fadeIn": {"value": 0.2}, "fadeOut": {"value": 0.3}},
    }

    manifest = build_manifest(job)

    assert manifest.schema_version == 3
    assert (manifest.logical_canvas.width, manifest.logical_canvas.height) == (320, 180)
    rendered = manifest.visual_items[0]
    assert rendered.style.preset == "Cinematic"
    assert rendered.style.adjustments.exposure == 10.4
    assert rendered.fades.fade_in_duration == 0.2
    assert _evaluated_opacity(rendered, 0.1) == pytest.approx(0.5)


def test_logical_canvas_scales_pixel_offsets_into_output_dimensions() -> None:
    job = render_job()
    job.document_json["settings"]["width"] = 1920
    job.document_json["settings"]["height"] = 1080
    job.document_json["tracks"][0]["items"][0]["transform"]["x"] = 192
    item = build_manifest(job).visual_items[0]
    source = Image.new("RGBA", (320, 180), (255, 255, 255, 255))

    layer, position = _render_visual_layer(source, item, 0, 320, 180, 1920, 1080)

    assert position[0] + layer.width / 2 == pytest.approx(192, abs=1)
    assert position[1] + layer.height / 2 == pytest.approx(90, abs=1)


def test_linked_audio_item_owns_video_audio_and_unlinked_video_uses_embedded_audio() -> None:
    job = render_job()
    source = job.media_sources[0]
    source.kind = "video"
    source.content_type = "video/mp4"
    item = job.document_json["tracks"][0]["items"][0]
    item.update(
        {
            "type": "video",
            "sourceIn": 0,
            "sourceOut": 0.5,
            "speed": 1,
            "linkedGroupId": "linked-1",
        }
    )
    job.document_json["media"]["image-1"]["kind"] = "video"

    embedded = build_manifest(job)
    assert [audio.source_owner for audio in embedded.audio_items] == ["embedded-video"]

    linked_job = job.model_copy(deep=True)
    linked_job.document_json["tracks"].append(
        {
            "id": "a1",
            "kind": "audio",
            "hidden": False,
            "muted": False,
            "items": [
                {
                    "id": "audio-1",
                    "type": "audio",
                    "mediaId": "image-1",
                    "timelineStart": 0,
                    "duration": 0.5,
                    "sourceIn": 0,
                    "sourceOut": 0.5,
                    "volume": 1,
                    "muted": False,
                    "fades": {"fadeInDuration": 0, "fadeOutDuration": 0},
                    "linkedGroupId": "linked-1",
                }
            ],
        }
    )
    linked = build_manifest(linked_job)
    assert len(linked.audio_items) == 1
    assert linked.audio_items[0].source_owner == "audio-item"
    assert linked.audio_items[0].linked_group_id == "linked-1"


async def test_audio_mix_covers_speed_fades_overlap_delay_gaps_and_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = video_render_job("mp4", "video/mp4")
    video_item = job.document_json["tracks"][0]["items"][0]
    video_item.update({"duration": 0.5, "sourceOut": 1, "speed": 2})
    job.document_json["tracks"].append(
        {
            "id": "a1",
            "kind": "audio",
            "hidden": False,
            "muted": False,
            "items": [
                {
                    "id": "audio-overlap",
                    "type": "audio",
                    "mediaId": "image-1",
                    "timelineStart": 1,
                    "duration": 0.5,
                    "sourceIn": 0,
                    "sourceOut": 0.5,
                    "volume": 0.5,
                    "muted": False,
                    "fades": {"fadeInDuration": 0.1, "fadeOutDuration": 0.2},
                }
            ],
        }
    )
    manifest = build_manifest(job)
    captured: list[list[str]] = []

    async def fake_probe(_path: Path) -> dict[str, object]:
        return {"streams": [{"codec_type": "audio"}]}

    async def fake_run(args: list[str]) -> None:
        captured.append(args)

    monkeypatch.setattr(ffmpeg, "ffprobe_json", fake_probe)
    monkeypatch.setattr(ffmpeg, "run_command", fake_run)
    source_path = tmp_path / "source.mp4"
    resolved = {
        "image-1": _ResolvedSource(source=job.media_sources[0], local_path=source_path),
    }

    assert await _render_audio_track(manifest, resolved, tmp_path / "audio.m4a") is True
    filter_complex = captured[0][captured[0].index("-filter_complex") + 1]
    assert "atempo=2.000000" in filter_complex
    assert "afade=t=in:st=0:d=0.100000" in filter_complex
    assert "afade=t=out:st=0.300000:d=0.200000" in filter_complex
    assert "adelay=1000|1000" in filter_complex
    assert "amix=inputs=2:duration=longest:dropout_transition=0" in filter_complex
    assert "apad=whole_dur=1.500000" in filter_complex


@pytest.mark.parametrize(
    "properties",
    [
        {"mask": {"shape": {"value": "Circle"}}},
        {"color": {"vignette": {"value": 25}}},
        {"speedSettings": {"reverse": {"value": True}}},
        {"audioSettings": {"limiter": {"value": True}}},
    ],
)
def test_non_default_unsupported_controls_block_render(properties: dict[str, object]) -> None:
    job = render_job()
    item = job.document_json["tracks"][0]["items"][0]
    item["properties"] = properties
    if "audioSettings" in properties:
        item.update({"type": "video", "sourceIn": 0, "sourceOut": 0.5, "speed": 1})

    with pytest.raises(RenderManifestError, match="not supported by export"):
        build_manifest(job)


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


def video_render_job(output_format: str, content_type: str) -> RenderJob:
    job = render_job()
    job.settings.update(
        {
            "format": output_format,
            "resolution": "current",
            "width": 160,
            "height": 90,
            "frameRate": 24,
        }
    )
    job.document_json["settings"].update({"width": 160, "height": 90})
    media = job.document_json["media"]["image-1"]
    media.update({"kind": "video", "width": 160, "height": 90})
    item = job.document_json["tracks"][0]["items"][0]
    item.update(
        {
            "type": "video",
            "duration": 0.5,
            "sourceIn": 0,
            "sourceOut": 0.5,
            "speed": 1,
        }
    )
    source = job.media_sources[0]
    source.kind = "video"
    source.object_key = "media/video/canonical.mp4"
    source.content_type = "video/mp4"
    source.duration_seconds = 0.6
    source.width = 160
    source.height = 90
    source.frame_rate = 24
    source.codec = "h264"
    job.output_storage_key = f"renders/job-1.{output_format}"
    job.output_content_type = content_type
    job.output_format = "mov" if output_format == "mov" else "mp4"
    return job
