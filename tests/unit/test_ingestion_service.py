"""Ingestion service MVP 1 orchestration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import kuvox_ai.modules.ingestion.service as service_module
from kuvox_ai.modules.ingestion import IngestionService
from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import DetectedShot, IngestionRequested, VideoMetadata
from kuvox_ai.modules.ingestion.ocr import ShotOcrText
from kuvox_ai.modules.ingestion.transcript import TranscriptSegment


def requested_payload() -> dict[str, object]:
    return {
        "eventId": "evt-1",
        "eventType": "ingestion.requested",
        "occurredAt": "2026-07-02T08:00:00Z",
        "mediaId": "media-1",
        "ownerId": "owner-1",
        "ownerKind": "User",
        "kind": "Video",
        "canonical": {
            "bucketName": "kuvox-canonical",
            "objectKey": "media/media-1/canonical.mp4",
            "contentType": "video/mp4",
            "sizeBytes": 456,
        },
        "durationSeconds": 12.0,
        "width": 1920,
        "height": 1080,
        "frameRate": 30.0,
        "codec": "h265",
    }


async def test_ingest_downloads_canonical_detects_shots_and_writes_graph(
    mock_kuzu: AsyncMock,
    mock_qdrant: AsyncMock,
    mock_storage: AsyncMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IngestionRequested.model_validate(requested_payload())
    writer = AsyncMock()
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=12,
        duration_seconds=12,
    )

    probe = AsyncMock(return_value=VideoMetadata(duration_seconds=12, width=1280, height=720))
    detect = AsyncMock(return_value=[shot])
    monkeypatch.setattr(service_module, "probe_video_metadata", probe)
    monkeypatch.setattr(service_module, "detect_video_shots", detect)
    events: list[str] = []
    frame = SampledFrame(shot=shot, timestamp_seconds=6.0, path=tmp_path / "frame.jpg")
    frame_sampler = AsyncMock()
    visual_encoder = AsyncMock()
    visual_writer = AsyncMock()
    audio_extractor = AsyncMock()
    transcriber = AsyncMock()
    text_encoder = AsyncMock()
    audio_encoder = AsyncMock()
    ocr_reader = AsyncMock()
    transcript_writer = AsyncMock()
    audio_writer = AsyncMock()
    ocr_writer = AsyncMock()
    audio_clip = ShotAudioClip(
        shot=shot,
        path=tmp_path / "shot.wav",
        clip_start_seconds=0.0,
        clip_end_seconds=12.0,
        clip_duration_seconds=12.0,
    )
    ocr_text = ShotOcrText(
        shot=shot,
        text="title card",
        frame_timestamp_seconds=6.0,
        text_block_count=1,
        mean_confidence=0.9,
    )

    def record_graph(*_: object) -> None:
        events.append("graph")

    def record_sample(*_: object) -> list[SampledFrame]:
        events.append("sample")
        return [frame]

    def record_encode(*_: object) -> list[list[float]]:
        events.append("encode")
        return [[0.0] * 512]

    def record_qdrant(*_: object) -> None:
        events.append("qdrant")

    def record_has_audio(*_: object) -> bool:
        events.append("has-audio")
        return True

    def record_full_audio(*_: object) -> Path:
        events.append("full-audio")
        return tmp_path / "full.wav"

    def record_transcribe(*_: object) -> list[TranscriptSegment]:
        events.append("transcribe")
        return [TranscriptSegment(start_seconds=0.0, end_seconds=4.0, text="hello")]

    def record_transcript_qdrant(*_: object) -> None:
        events.append("transcript-qdrant")

    def record_shot_audio(*_: object) -> list[ShotAudioClip]:
        events.append("shot-audio")
        return [audio_clip]

    def record_audio_encode(*_: object) -> list[list[float]]:
        events.append("audio-encode")
        return [[0.3] * 1024]

    def record_audio_qdrant(*_: object) -> None:
        events.append("audio-qdrant")

    def record_ocr(*_: object) -> list[ShotOcrText]:
        events.append("ocr")
        return [ocr_text]

    def record_ocr_qdrant(*_: object) -> None:
        events.append("ocr-qdrant")

    writer.write_video_with_shots.side_effect = record_graph
    frame_sampler.sample_frames.side_effect = record_sample
    visual_encoder.encode_frames.side_effect = record_encode
    visual_writer.write_shot_vectors.side_effect = record_qdrant
    audio_extractor.has_audio_stream.side_effect = record_has_audio
    audio_extractor.extract_full_audio.side_effect = record_full_audio
    transcriber.transcribe.side_effect = record_transcribe
    text_encoder.encode_texts.side_effect = [
        [[0.1] * 384],
        [[0.2] * 384],
    ]
    transcript_writer.write_points.side_effect = record_transcript_qdrant
    audio_extractor.extract_shot_audio_clips.side_effect = record_shot_audio
    audio_encoder.encode_audio_clips.side_effect = record_audio_encode
    audio_writer.write_points.side_effect = record_audio_qdrant
    ocr_reader.read_frames.side_effect = record_ocr
    ocr_writer.write_points.side_effect = record_ocr_qdrant

    svc = IngestionService(
        kuzu=mock_kuzu,
        qdrant=mock_qdrant,
        storage=mock_storage,
        work_dir=tmp_path,
        writer=writer,
        frame_sampler=frame_sampler,
        visual_encoder=visual_encoder,
        visual_writer=visual_writer,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        ocr_reader=ocr_reader,
        transcript_writer=transcript_writer,
        audio_writer=audio_writer,
        ocr_writer=ocr_writer,
    )

    result = await svc.ingest(request)

    mock_storage.download_file.assert_awaited_once()
    assert mock_storage.download_file.await_args.args[:2] == (
        "kuvox-canonical",
        "media/media-1/canonical.mp4",
    )
    assert mock_storage.download_file.await_args.args[2].name == "canonical.mp4"
    detect.assert_awaited_once()
    writer.write_video_with_shots.assert_awaited_once()
    frame_sampler.sample_frames.assert_awaited_once()
    visual_encoder.encode_frames.assert_awaited_once_with([frame])
    visual_writer.write_shot_vectors.assert_awaited_once_with(request, [frame], [[0.0] * 512])
    transcript_writer.write_points.assert_awaited_once()
    audio_writer.write_points.assert_awaited_once()
    ocr_writer.write_points.assert_awaited_once()
    assert events == [
        "graph",
        "sample",
        "encode",
        "qdrant",
        "has-audio",
        "full-audio",
        "transcribe",
        "transcript-qdrant",
        "shot-audio",
        "audio-encode",
        "audio-qdrant",
        "ocr",
        "ocr-qdrant",
    ]
    assert result.source_event_id == "evt-1"
    assert result.media_id == "media-1"
    assert result.shot_count == 1


async def test_ingest_visual_indexing_errors_do_not_block_completion(
    mock_kuzu: AsyncMock,
    mock_storage: AsyncMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IngestionRequested.model_validate(requested_payload())
    writer = AsyncMock()
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=12,
        duration_seconds=12,
    )
    frame = SampledFrame(shot=shot, timestamp_seconds=6.0, path=tmp_path / "frame.jpg")

    monkeypatch.setattr(
        service_module,
        "probe_video_metadata",
        AsyncMock(return_value=VideoMetadata(duration_seconds=12)),
    )
    monkeypatch.setattr(service_module, "detect_video_shots", AsyncMock(return_value=[shot]))
    frame_sampler = AsyncMock()
    visual_encoder = AsyncMock()
    visual_writer = AsyncMock()
    audio_extractor = AsyncMock()
    transcriber = AsyncMock()
    text_encoder = AsyncMock()
    audio_encoder = AsyncMock()
    ocr_reader = AsyncMock()
    transcript_writer = AsyncMock()
    audio_writer = AsyncMock()
    ocr_writer = AsyncMock()
    frame_sampler.sample_frames.return_value = [frame]
    visual_encoder.encode_frames.return_value = [[0.0] * 512]
    visual_writer.write_shot_vectors.side_effect = RuntimeError("qdrant down")
    audio_extractor.has_audio_stream.return_value = False
    ocr_reader.read_frames.return_value = []
    text_encoder.encode_texts.return_value = []

    svc = IngestionService(
        kuzu=mock_kuzu,
        storage=mock_storage,
        work_dir=tmp_path,
        writer=writer,
        frame_sampler=frame_sampler,
        visual_encoder=visual_encoder,
        visual_writer=visual_writer,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        ocr_reader=ocr_reader,
        transcript_writer=transcript_writer,
        audio_writer=audio_writer,
        ocr_writer=ocr_writer,
    )

    result = await svc.ingest(request)

    writer.write_video_with_shots.assert_awaited_once()
    visual_writer.write_shot_vectors.assert_awaited_once()
    transcript_writer.write_points.assert_awaited_once_with(request, [])
    audio_writer.write_points.assert_awaited_once_with(request, [])
    ocr_writer.write_points.assert_awaited_once()
    assert result.media_id == "media-1"


async def test_ingest_no_audio_deletes_transcript_audio_points_and_indexes_ocr(
    mock_kuzu: AsyncMock,
    mock_storage: AsyncMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IngestionRequested.model_validate(requested_payload())
    writer = AsyncMock()
    shot = DetectedShot(
        shot_id="media-1:shot:000000",
        media_id="media-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=12,
        duration_seconds=12,
    )
    frame = SampledFrame(shot=shot, timestamp_seconds=6.0, path=tmp_path / "frame.jpg")
    ocr_text = ShotOcrText(
        shot=shot,
        text="onscreen",
        frame_timestamp_seconds=6.0,
        text_block_count=1,
        mean_confidence=0.8,
    )

    monkeypatch.setattr(
        service_module,
        "probe_video_metadata",
        AsyncMock(return_value=VideoMetadata(duration_seconds=12)),
    )
    monkeypatch.setattr(service_module, "detect_video_shots", AsyncMock(return_value=[shot]))
    frame_sampler = AsyncMock()
    visual_encoder = AsyncMock()
    visual_writer = AsyncMock()
    audio_extractor = AsyncMock()
    transcriber = AsyncMock()
    text_encoder = AsyncMock()
    audio_encoder = AsyncMock()
    ocr_reader = AsyncMock()
    transcript_writer = AsyncMock()
    audio_writer = AsyncMock()
    ocr_writer = AsyncMock()
    frame_sampler.sample_frames.return_value = [frame]
    visual_encoder.encode_frames.return_value = [[0.0] * 512]
    audio_extractor.has_audio_stream.return_value = False
    ocr_reader.read_frames.return_value = [ocr_text]
    text_encoder.encode_texts.return_value = [[0.2] * 384]

    svc = IngestionService(
        kuzu=mock_kuzu,
        storage=mock_storage,
        work_dir=tmp_path,
        writer=writer,
        frame_sampler=frame_sampler,
        visual_encoder=visual_encoder,
        visual_writer=visual_writer,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        ocr_reader=ocr_reader,
        transcript_writer=transcript_writer,
        audio_writer=audio_writer,
        ocr_writer=ocr_writer,
    )

    await svc.ingest(request)

    transcript_writer.write_points.assert_awaited_once_with(request, [])
    audio_writer.write_points.assert_awaited_once_with(request, [])
    transcriber.transcribe.assert_not_awaited()
    audio_encoder.encode_audio_clips.assert_not_awaited()
    ocr_writer.write_points.assert_awaited_once()
