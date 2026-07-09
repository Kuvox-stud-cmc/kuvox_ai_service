"""Build Qdrant points for non-visual shot modalities."""

from __future__ import annotations

from kuvox_ai.modules.ingestion.audio import ShotAudioClip
from kuvox_ai.modules.ingestion.models import (
    AudioMetadata,
    DetectedShot,
    ImageMetadata,
    IngestionRequested,
)
from kuvox_ai.modules.ingestion.ocr import ShotOcrText
from kuvox_ai.modules.ingestion.qdrant_writer import ShotVectorPoint
from kuvox_ai.modules.ingestion.transcript import ShotTranscript


def transcript_points(
    request: IngestionRequested,
    transcripts: list[ShotTranscript],
    embeddings: list[list[float]],
) -> list[ShotVectorPoint]:
    if len(transcripts) != len(embeddings):
        raise ValueError(
            f"Expected {len(transcripts)} transcript embeddings, got {len(embeddings)}."
        )
    return [
        ShotVectorPoint(
            point_id=transcript.shot.shot_id,
            vector=embedding,
            payload={
                **common_payload(request, transcript.shot, "transcript"),
                "text": transcript.text,
                "segmentCount": transcript.segment_count,
            },
        )
        for transcript, embedding in zip(transcripts, embeddings, strict=True)
        if transcript.text.strip()
    ]


def ocr_points(
    request: IngestionRequested,
    ocr_texts: list[ShotOcrText],
    embeddings: list[list[float]],
) -> list[ShotVectorPoint]:
    if len(ocr_texts) != len(embeddings):
        raise ValueError(f"Expected {len(ocr_texts)} OCR embeddings, got {len(embeddings)}.")
    return [
        ShotVectorPoint(
            point_id=ocr_text.shot.shot_id,
            vector=embedding,
            payload={
                **common_payload(request, ocr_text.shot, "ocr"),
                "text": ocr_text.text,
                "frameTimestampSeconds": ocr_text.frame_timestamp_seconds,
                "textBlockCount": ocr_text.text_block_count,
                "meanConfidence": ocr_text.mean_confidence,
            },
        )
        for ocr_text, embedding in zip(ocr_texts, embeddings, strict=True)
        if ocr_text.text.strip()
    ]


def audio_points(
    request: IngestionRequested,
    clips: list[ShotAudioClip],
    embeddings: list[list[float]],
) -> list[ShotVectorPoint]:
    if len(clips) != len(embeddings):
        raise ValueError(f"Expected {len(clips)} audio embeddings, got {len(embeddings)}.")
    return [
        ShotVectorPoint(
            point_id=clip.shot.shot_id,
            vector=embedding,
            payload={
                **common_payload(request, clip.shot, "audio"),
                "clipStartSeconds": clip.clip_start_seconds,
                "clipEndSeconds": clip.clip_end_seconds,
                "clipDurationSeconds": clip.clip_duration_seconds,
            },
        )
        for clip, embedding in zip(clips, embeddings, strict=True)
    ]


def image_visual_point(
    request: IngestionRequested,
    metadata: ImageMetadata,
    embedding: list[float],
) -> ShotVectorPoint:
    return ShotVectorPoint(
        point_id=f"{request.media_id}:image",
        vector=embedding,
        payload={
            **media_payload(request, "visual"),
            "width": metadata.width,
            "height": metadata.height,
        },
    )


def image_ocr_point(
    request: IngestionRequested,
    text: str,
    embedding: list[float],
    *,
    text_block_count: int,
    mean_confidence: float | None,
) -> ShotVectorPoint:
    return ShotVectorPoint(
        point_id=f"{request.media_id}:ocr",
        vector=embedding,
        payload={
            **media_payload(request, "ocr"),
            "text": text,
            "textBlockCount": text_block_count,
            "meanConfidence": mean_confidence,
        },
    )


def audio_media_point(
    request: IngestionRequested,
    metadata: AudioMetadata,
    embedding: list[float],
) -> ShotVectorPoint:
    return ShotVectorPoint(
        point_id=f"{request.media_id}:audio",
        vector=embedding,
        payload={
            **media_payload(request, "audio"),
            "durationSeconds": metadata.duration_seconds,
            "codec": metadata.codec,
        },
    )


def audio_transcript_point(
    request: IngestionRequested,
    text: str,
    embedding: list[float],
    *,
    segment_count: int,
) -> ShotVectorPoint:
    return ShotVectorPoint(
        point_id=f"{request.media_id}:transcript",
        vector=embedding,
        payload={
            **media_payload(request, "transcript"),
            "text": text,
            "segmentCount": segment_count,
        },
    )


def common_payload(
    request: IngestionRequested,
    shot: DetectedShot,
    modality: str,
) -> dict[str, object]:
    return {
        "mediaId": request.media_id,
        "ownerId": request.owner_id,
        "ownerKind": request.owner_kind.value,
        "shotId": shot.shot_id,
        "shotIndex": shot.shot_index,
        "startSeconds": shot.start_seconds,
        "endSeconds": shot.end_seconds,
        "durationSeconds": shot.duration_seconds,
        "modality": modality,
    }


def media_payload(request: IngestionRequested, modality: str) -> dict[str, object]:
    return {
        "mediaId": request.media_id,
        "ownerId": request.owner_id,
        "ownerKind": request.owner_kind.value,
        "kind": request.kind.value,
        "canonicalObjectKey": request.canonical.object_key,
        "contentType": request.canonical.content_type,
        "modality": modality,
    }
