from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest

from kuvox_ai.cache import CacheRead, CacheStore, ReadOutcome, WriteOutcome
from kuvox_ai.modules.ingestion import IngestionService
from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.audio_embedding_cache import CachedAudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.audio_encoder import AudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.frame_sampler import SampledFrame
from kuvox_ai.modules.ingestion.models import (
    AudioMetadata,
    DetectedShot,
    ImageMetadata,
    IngestionRequested,
)
from kuvox_ai.modules.ingestion.ocr import ShotOcrText
from kuvox_ai.modules.ingestion.service import media_level_shot
from kuvox_ai.modules.ingestion.text_embedding_cache import CachedIngestionTextEmbeddingEncoder
from kuvox_ai.modules.ingestion.text_encoder import TextEmbeddingEncoder
from kuvox_ai.modules.ingestion.transcript import TranscriptSegment
from kuvox_ai.modules.ingestion.visual_embedding_cache import CachedVisualEmbeddingEncoder
from kuvox_ai.modules.ingestion.visual_encoder import VisualEncoder


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str) -> CacheRead:
        value = self.values.get(key)
        return (
            CacheRead(ReadOutcome.HIT, value) if value is not None else CacheRead(ReadOutcome.MISS)
        )

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> WriteOutcome:
        del ttl_seconds
        self.values[key] = value
        return WriteOutcome.SUCCESS

    async def delete(self, key: str) -> WriteOutcome:
        self.values.pop(key, None)
        return WriteOutcome.SUCCESS


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1] * 384 for _ in texts]


class RecordingVisualEncoder:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    async def encode_frames(self, frames: list[SampledFrame]) -> list[list[float]]:
        self.calls.append([frame.path for frame in frames])
        return [[0.25] * 512 for _ in frames]


class RecordingAudioEncoder:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    async def encode_audio_clips(self, clips: list[ShotAudioClip]) -> list[list[float]]:
        self.calls.append([clip.path for clip in clips])
        return [[-0.5] * 1024 for _ in clips]


@pytest.mark.asyncio
async def test_all_four_ingestion_text_consumers_share_cached_vectors(tmp_path: Path) -> None:
    video_request = request("video-1", "Video", "video/mp4")
    audio_request = request("audio-1", "Audio", "audio/opus")
    image_request = request("image-1", "Image", "image/webp")
    video_shot = DetectedShot(
        shot_id="video-1:shot:000000",
        media_id="video-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=4,
        duration_seconds=4,
    )
    video_frame = SampledFrame(
        shot=video_shot,
        timestamp_seconds=2,
        path=tmp_path / "video.jpg",
    )
    image_frame = SampledFrame(
        shot=media_level_shot(image_request),
        timestamp_seconds=0,
        path=tmp_path / "image.jpg",
    )
    shared_text = "Anonymized launch timeline"

    transcriber = AsyncMock()
    transcriber.transcribe.side_effect = [
        [TranscriptSegment(0, 4, shared_text)],
        [TranscriptSegment(0, 4, shared_text)],
    ]
    ocr_reader = AsyncMock()
    ocr_reader.read_frames.side_effect = [
        [ShotOcrText(video_shot, shared_text, 2, 1, 0.9)],
        [ShotOcrText(image_frame.shot, shared_text, 0, 1, 0.9)],
    ]
    audio_extractor = AsyncMock()
    audio_extractor.has_audio_stream.return_value = True
    audio_extractor.extract_full_audio.return_value = tmp_path / "full.wav"
    audio_extractor.extract_shot_audio_clips.return_value = []
    audio_encoder = AsyncMock()
    audio_encoder.encode_audio_clips.return_value = []
    authoritative = RecordingEncoder()
    cached_encoder = CachedIngestionTextEmbeddingEncoder(
        cast(TextEmbeddingEncoder, authoritative),
        cache=cast(CacheStore, MemoryCache()),
        enabled=True,
        model_id="model/shared",
        dimension=384,
        legacy_read_enabled=False,
    )
    transcript_writer = AsyncMock()
    audio_writer = AsyncMock()
    ocr_writer = AsyncMock()
    media_transcript_writer = AsyncMock()
    media_ocr_writer = AsyncMock()
    service = IngestionService(
        kuzu=AsyncMock(),
        storage=AsyncMock(),
        work_dir=tmp_path,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        text_encoder=cached_encoder,
        audio_encoder=audio_encoder,
        ocr_reader=ocr_reader,
        transcript_writer=transcript_writer,
        audio_writer=audio_writer,
        ocr_writer=ocr_writer,
        media_transcript_writer=media_transcript_writer,
        media_ocr_writer=media_ocr_writer,
    )

    await service._index_transcript_audio_ocr(
        request=video_request,
        canonical_path=tmp_path / "video.mp4",
        shots=[video_shot],
        frames=[video_frame],
        job_dir=tmp_path,
    )
    await service._index_audio_transcript(audio_request, tmp_path / "audio.opus")
    await service._index_image_ocr(image_request, image_frame)

    assert authoritative.calls == [[shared_text]]
    transcript = transcript_writer.write_points.await_args.args[1][0]
    shot_ocr = ocr_writer.write_points.await_args.args[1][0]
    media_transcript = media_transcript_writer.write_points.await_args.args[1][0]
    media_ocr = media_ocr_writer.write_points.await_args.args[1][0]
    assert [
        transcript.payload["text"],
        shot_ocr.payload["text"],
        media_transcript.payload["text"],
        media_ocr.payload["text"],
    ] == [shared_text] * 4
    for point in (transcript, shot_ocr, media_transcript, media_ocr):
        assert point.vector == pytest.approx([0.1] * 384)


@pytest.mark.asyncio
async def test_all_four_file_embedding_consumers_reuse_vectors_in_qdrant_dtos(
    tmp_path: Path,
) -> None:
    video_request = request("video-1", "Video", "video/mp4")
    image_request = request("image-1", "Image", "image/png")
    audio_request = request("audio-1", "Audio", "audio/wav")
    video_shot = DetectedShot(
        shot_id="video-1:shot:000000",
        media_id="video-1",
        shot_index=0,
        start_seconds=0,
        end_seconds=1,
        duration_seconds=1,
    )
    shot_frame_path = tmp_path / "shot.png"
    image_path = tmp_path / "image.png"
    shot_audio_path = tmp_path / "shot.wav"
    media_audio_path = tmp_path / "media.wav"
    shot_frame_path.write_bytes(b"shared png bytes")
    image_path.write_bytes(b"shared png bytes")
    shot_audio_path.write_bytes(b"shared wav bytes")
    media_audio_path.write_bytes(b"shared wav bytes")
    shot_frame = SampledFrame(video_shot, 0.5, shot_frame_path)
    image_frame = SampledFrame(media_level_shot(image_request), 0.0, image_path)
    shot_clip = ShotAudioClip(video_shot, shot_audio_path, 0.0, 1.0, 1.0)

    store = cast(CacheStore, MemoryCache())
    visual_authoritative = RecordingVisualEncoder()
    audio_authoritative = RecordingAudioEncoder()
    visual = CachedVisualEmbeddingEncoder(
        cast(VisualEncoder, visual_authoritative),
        cache=store,
        enabled=True,
        model_id="openclip:test",
        dimension=512,
    )
    audio = CachedAudioEmbeddingEncoder(
        cast(AudioEmbeddingEncoder, audio_authoritative),
        cache=store,
        enabled=True,
        model_id="msclap:test",
        dimension=1024,
    )
    visual_writer = AsyncMock()
    media_visual_writer = AsyncMock()
    audio_writer = AsyncMock()
    media_audio_writer = AsyncMock()
    audio_extractor = AsyncMock()
    audio_extractor.has_audio_stream.return_value = True
    audio_extractor.extract_full_audio.return_value = shot_audio_path
    audio_extractor.extract_shot_audio_clips.return_value = [shot_clip]
    transcriber = AsyncMock()
    transcriber.transcribe.return_value = []
    text_encoder = AsyncMock()
    text_encoder.encode_texts.return_value = []
    ocr_reader = AsyncMock()
    ocr_reader.read_frames.return_value = []
    service = IngestionService(
        kuzu=AsyncMock(),
        storage=AsyncMock(),
        work_dir=tmp_path,
        visual_encoder=visual,
        audio_encoder=audio,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        text_encoder=text_encoder,
        ocr_reader=ocr_reader,
        visual_writer=visual_writer,
        media_visual_writer=media_visual_writer,
        transcript_writer=AsyncMock(),
        audio_writer=audio_writer,
        ocr_writer=AsyncMock(),
        media_audio_writer=media_audio_writer,
    )

    await service._index_visual(video_request, [shot_frame])
    await service._index_media_visual(
        image_request,
        image_frame,
        ImageMetadata(width=64, height=64),
    )
    await service._index_transcript_audio_ocr(
        request=video_request,
        canonical_path=tmp_path / "video.mp4",
        shots=[video_shot],
        frames=[shot_frame],
        job_dir=tmp_path,
    )
    await service._index_media_audio(
        audio_request,
        media_audio_path,
        AudioMetadata(duration_seconds=1.0, codec="pcm_s16le"),
    )

    assert visual_authoritative.calls == [[shot_frame_path]]
    assert audio_authoritative.calls == [[shot_audio_path]]
    shot_visual_vector = visual_writer.write_shot_vectors.await_args.args[2][0]
    image_visual_point = media_visual_writer.write_points.await_args.args[1][0]
    shot_audio_point = audio_writer.write_points.await_args.args[1][0]
    media_audio_point = media_audio_writer.write_points.await_args.args[1][0]
    assert shot_visual_vector == image_visual_point.vector == [0.25] * 512
    assert shot_audio_point.vector == media_audio_point.vector == [-0.5] * 1024
    assert image_visual_point.payload["modality"] == "visual"
    assert shot_audio_point.payload["modality"] == "audio"
    assert media_audio_point.payload["modality"] == "audio"


def request(media_id: str, kind: str, content_type: str) -> IngestionRequested:
    return IngestionRequested.model_validate(
        {
            "eventId": f"event-{media_id}",
            "eventType": "ingestion.requested",
            "occurredAt": "2026-07-17T00:00:00Z",
            "mediaId": media_id,
            "ownerId": "anonymized-owner",
            "ownerKind": "User",
            "kind": kind,
            "canonical": {
                "bucketName": "kuvox-canonical",
                "objectKey": f"evidence/{media_id}",
                "contentType": content_type,
                "sizeBytes": 1,
            },
        }
    )
