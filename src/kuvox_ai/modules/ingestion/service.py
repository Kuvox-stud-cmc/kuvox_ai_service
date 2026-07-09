"""Ingestion service for video graph writes and visual shot indexing."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from kuvox_ai.infrastructure import KuzuClient, ObjectStorageClient, QdrantClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.ingestion.audio import AudioExtractor, FFmpegAudioExtractor, ShotAudioClip
from kuvox_ai.modules.ingestion.audio_encoder import AudioEmbeddingEncoder, MsClapAudioEncoder
from kuvox_ai.modules.ingestion.frame_sampler import FFmpegFrameSampler, FrameSampler, SampledFrame
from kuvox_ai.modules.ingestion.kuzu_writer import KuzuIngestionWriter
from kuvox_ai.modules.ingestion.metadata import (
    probe_audio_metadata,
    probe_image_metadata,
    probe_video_metadata,
)
from kuvox_ai.modules.ingestion.modality_points import (
    audio_media_point,
    audio_points,
    audio_transcript_point,
    image_ocr_point,
    image_visual_point,
    ocr_points,
    transcript_points,
)
from kuvox_ai.modules.ingestion.models import (
    AudioMetadata,
    DetectedShot,
    ImageMetadata,
    IngestionCompleted,
    IngestionRequested,
    MediaKind,
    VideoMetadata,
)
from kuvox_ai.modules.ingestion.ocr import EasyOcrReader, OcrReader
from kuvox_ai.modules.ingestion.qdrant_writer import (
    QdrantShotEmbeddingWriter,
    QdrantShotVisualWriter,
    ShotEmbeddingIndexWriter,
    VisualIndexWriter,
)
from kuvox_ai.modules.ingestion.shot_detection import detect_video_shots
from kuvox_ai.modules.ingestion.text_encoder import (
    SentenceTransformerTextEncoder,
    TextEmbeddingEncoder,
)
from kuvox_ai.modules.ingestion.transcript import (
    FasterWhisperTranscriber,
    Transcriber,
    align_transcript_to_shots,
)
from kuvox_ai.modules.ingestion.visual_encoder import ClipVisualEncoder, VisualEncoder

logger = get_logger(__name__)


class IngestionService:
    """Download canonical video, detect shots, write graph data, and index frames."""

    def __init__(
        self,
        *,
        kuzu: KuzuClient,
        storage: ObjectStorageClient,
        work_dir: Path,
        qdrant: QdrantClient | None = None,
        writer: KuzuIngestionWriter | None = None,
        frame_sampler: FrameSampler | None = None,
        visual_encoder: VisualEncoder | None = None,
        visual_writer: VisualIndexWriter | None = None,
        audio_extractor: AudioExtractor | None = None,
        transcriber: Transcriber | None = None,
        text_encoder: TextEmbeddingEncoder | None = None,
        audio_encoder: AudioEmbeddingEncoder | None = None,
        ocr_reader: OcrReader | None = None,
        transcript_writer: ShotEmbeddingIndexWriter | None = None,
        audio_writer: ShotEmbeddingIndexWriter | None = None,
        ocr_writer: ShotEmbeddingIndexWriter | None = None,
        media_visual_writer: ShotEmbeddingIndexWriter | None = None,
        media_audio_writer: ShotEmbeddingIndexWriter | None = None,
        media_transcript_writer: ShotEmbeddingIndexWriter | None = None,
        media_ocr_writer: ShotEmbeddingIndexWriter | None = None,
        visual_collection_name: str = "shots_visual",
        visual_embedding_dim: int = 512,
        transcript_collection_name: str = "shots_transcript",
        audio_collection_name: str = "shots_audio",
        ocr_collection_name: str = "shots_ocr",
        media_visual_collection_name: str = "media_visual",
        media_audio_collection_name: str = "media_audio",
        media_transcript_collection_name: str = "media_transcript",
        media_ocr_collection_name: str = "media_ocr",
        text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_embedding_dim: int = 384,
        text_embedding_device: str = "auto",
        text_embedding_batch_size: int = 32,
        whisper_model_name: str = "small",
        whisper_device: str = "auto",
        whisper_compute_type: str = "auto",
        audio_embedding_dim: int = 1024,
        audio_embedding_device: str = "auto",
        audio_embedding_batch_size: int = 16,
        ocr_languages: list[str] | None = None,
        ocr_gpu: str = "auto",
        ocr_min_confidence: float = 0.3,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        clip_device: str = "auto",
        clip_batch_size: int = 16,
        visual_index_timeout_seconds: int = 60,
        optional_index_timeout_seconds: int = 60,
    ) -> None:
        self._kuzu = kuzu
        self._storage = storage
        self._work_dir = work_dir
        self._qdrant = qdrant
        self._visual_index_timeout_seconds = visual_index_timeout_seconds
        self._optional_index_timeout_seconds = optional_index_timeout_seconds
        self._writer = writer or KuzuIngestionWriter(kuzu)
        self._frame_sampler = frame_sampler or FFmpegFrameSampler()
        self._visual_encoder = visual_encoder or ClipVisualEncoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=clip_device,
            batch_size=clip_batch_size,
        )
        self._audio_extractor = audio_extractor or FFmpegAudioExtractor()
        self._transcriber = transcriber or FasterWhisperTranscriber(
            model_name=whisper_model_name,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        self._text_encoder = text_encoder or SentenceTransformerTextEncoder(
            model_name=text_embedding_model_name,
            device=text_embedding_device,
            batch_size=text_embedding_batch_size,
        )
        self._audio_encoder = audio_encoder or MsClapAudioEncoder(
            device=audio_embedding_device,
            embedding_dim=audio_embedding_dim,
            batch_size=audio_embedding_batch_size,
        )
        self._ocr_reader = ocr_reader or EasyOcrReader(
            languages=ocr_languages or ["en"],
            gpu=ocr_gpu,
            min_confidence=ocr_min_confidence,
        )
        self._visual_writer: VisualIndexWriter | None
        if visual_writer is not None:
            self._visual_writer = visual_writer
        elif qdrant is not None:
            self._visual_writer = QdrantShotVisualWriter(
                qdrant,
                collection_name=visual_collection_name,
                embedding_dim=visual_embedding_dim,
            )
        else:
            self._visual_writer = None

        self._transcript_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=transcript_writer,
            collection_name=transcript_collection_name,
            embedding_dim=text_embedding_dim,
        )
        self._audio_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=audio_writer,
            collection_name=audio_collection_name,
            embedding_dim=audio_embedding_dim,
        )
        self._ocr_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=ocr_writer,
            collection_name=ocr_collection_name,
            embedding_dim=text_embedding_dim,
        )
        self._media_visual_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=media_visual_writer,
            collection_name=media_visual_collection_name,
            embedding_dim=visual_embedding_dim,
        )
        self._media_audio_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=media_audio_writer,
            collection_name=media_audio_collection_name,
            embedding_dim=audio_embedding_dim,
        )
        self._media_transcript_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=media_transcript_writer,
            collection_name=media_transcript_collection_name,
            embedding_dim=text_embedding_dim,
        )
        self._media_ocr_writer = build_embedding_writer(
            qdrant=qdrant,
            writer=media_ocr_writer,
            collection_name=media_ocr_collection_name,
            embedding_dim=text_embedding_dim,
        )

    async def ingest(self, request: IngestionRequested) -> IngestionCompleted:
        if request.kind == MediaKind.video:
            return await self.ingest_video(request)
        if request.kind == MediaKind.audio:
            return await self.ingest_audio(request)
        if request.kind == MediaKind.image:
            return await self.ingest_image(request)

        raise ValueError(f"Unsupported media kind: {request.kind}")

    async def ingest_video(self, request: IngestionRequested) -> IngestionCompleted:
        if self._visual_writer is None:
            raise RuntimeError("Visual indexing requires a connected Qdrant writer.")
        if (
            self._transcript_writer is None
            or self._audio_writer is None
            or self._ocr_writer is None
        ):
            raise RuntimeError("Transcript, audio, and OCR indexing require Qdrant writers.")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ingestion.ingest.start", media_id=request.media_id)

        with temporary_job_dir(self._work_dir) as job_dir:
            canonical_path = job_dir / canonical_filename(request.canonical.object_key)
            await self._storage.download_file(
                request.canonical.bucket_name,
                request.canonical.object_key,
                canonical_path,
            )
            logger.info("ingestion.ingest.downloaded", media_id=request.media_id)

            metadata = merge_metadata(
                request_metadata=VideoMetadata(
                    duration_seconds=request.duration_seconds,
                    width=request.width,
                    height=request.height,
                    frame_rate=request.frame_rate,
                    codec=request.codec,
                ),
                probed_metadata=await probe_video_metadata(canonical_path),
            )
            logger.info("ingestion.ingest.metadata_ready", media_id=request.media_id)
            shots = await detect_video_shots(
                canonical_path,
                media_id=request.media_id,
                duration_seconds=metadata.duration_seconds or 0.0,
            )
            logger.info(
                "ingestion.ingest.shots_detected",
                media_id=request.media_id,
                shot_count=len(shots),
            )
            await self._writer.write_video_with_shots(request, metadata, shots)
            logger.info("ingestion.ingest.graph_written", media_id=request.media_id)
            frames = await self._frame_sampler.sample_frames(
                canonical_path,
                shots,
                job_dir / "frames",
            )
            logger.info(
                "ingestion.ingest.frames_sampled",
                media_id=request.media_id,
                frame_count=len(frames),
            )
            visual_count = await self._index_visual_best_effort(request, frames)
            optional_counts = await self._index_optional_best_effort(
                request,
                canonical_path,
                shots,
                frames,
                job_dir,
            )

        logger.info("ingestion.ingest.completed", media_id=request.media_id, shot_count=len(shots))
        return IngestionCompleted(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            shot_count=len(shots),
            visual_count=visual_count,
            audio_count=optional_counts["audio"],
            transcript_count=optional_counts["transcript"],
            ocr_count=optional_counts["ocr"],
        )

    async def ingest_audio(self, request: IngestionRequested) -> IngestionCompleted:
        if self._media_audio_writer is None:
            raise RuntimeError("Audio ingestion requires a connected media audio writer.")
        if self._media_transcript_writer is None:
            raise RuntimeError("Audio transcript ingestion requires a connected media writer.")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ingestion.audio.start", media_id=request.media_id)

        with temporary_job_dir(self._work_dir) as job_dir:
            canonical_path = job_dir / canonical_filename(request.canonical.object_key)
            await self._storage.download_file(
                request.canonical.bucket_name,
                request.canonical.object_key,
                canonical_path,
            )
            metadata = merge_audio_metadata(
                request_metadata=AudioMetadata(
                    duration_seconds=request.duration_seconds,
                    codec=request.codec,
                ),
                probed_metadata=await probe_audio_metadata(canonical_path),
            )
            await self._writer.write_audio(request, metadata)
            audio_count = await self._index_media_audio(request, canonical_path, metadata)
            transcript_count = await self._index_audio_transcript_best_effort(
                request,
                canonical_path,
            )

        logger.info("ingestion.audio.completed", media_id=request.media_id)
        return IngestionCompleted(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            shot_count=0,
            audio_count=audio_count,
            transcript_count=transcript_count,
        )

    async def ingest_image(self, request: IngestionRequested) -> IngestionCompleted:
        if self._media_visual_writer is None:
            raise RuntimeError("Image ingestion requires a connected media visual writer.")
        if self._media_ocr_writer is None:
            raise RuntimeError("Image OCR ingestion requires a connected media writer.")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ingestion.image.start", media_id=request.media_id)

        with temporary_job_dir(self._work_dir) as job_dir:
            canonical_path = job_dir / canonical_filename(request.canonical.object_key)
            await self._storage.download_file(
                request.canonical.bucket_name,
                request.canonical.object_key,
                canonical_path,
            )
            metadata = merge_image_metadata(
                request_metadata=ImageMetadata(width=request.width, height=request.height),
                probed_metadata=await probe_image_metadata(canonical_path),
            )
            await self._writer.write_image(request, metadata)
            frame = media_image_frame(request, canonical_path)
            visual_count = await self._index_media_visual(request, frame, metadata)
            ocr_count = await self._index_image_ocr_best_effort(request, frame)

        logger.info("ingestion.image.completed", media_id=request.media_id)
        return IngestionCompleted(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            source_event_id=request.event_id,
            media_id=request.media_id,
            shot_count=0,
            visual_count=visual_count,
            ocr_count=ocr_count,
        )

    async def _index_visual_best_effort(
        self,
        request: IngestionRequested,
        frames: list[SampledFrame],
    ) -> int:
        assert self._visual_writer is not None
        try:
            return await asyncio.wait_for(
                self._index_visual(request, frames),
                timeout=self._visual_index_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "ingestion.ingest.visual_index_timeout",
                media_id=request.media_id,
                timeout_seconds=self._visual_index_timeout_seconds,
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingestion.ingest.visual_index_failed",
                media_id=request.media_id,
                error=str(exc),
            )
            return 0

    async def _index_visual(
        self,
        request: IngestionRequested,
        frames: list[SampledFrame],
    ) -> int:
        assert self._visual_writer is not None
        embeddings = await self._visual_encoder.encode_frames(frames)
        await self._visual_writer.write_shot_vectors(request, frames, embeddings)
        logger.info("ingestion.ingest.visual_indexed", media_id=request.media_id)
        return len(embeddings)

    async def _index_optional_best_effort(
        self,
        request: IngestionRequested,
        canonical_path: Path,
        shots: list[DetectedShot],
        frames: list[SampledFrame],
        job_dir: Path,
    ) -> dict[str, int]:
        try:
            return await asyncio.wait_for(
                self._index_transcript_audio_ocr(
                    request=request,
                    canonical_path=canonical_path,
                    shots=shots,
                    frames=frames,
                    job_dir=job_dir,
                ),
                timeout=self._optional_index_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "ingestion.ingest.optional_index_timeout",
                media_id=request.media_id,
                timeout_seconds=self._optional_index_timeout_seconds,
            )
            return {"transcript": 0, "audio": 0, "ocr": 0}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingestion.ingest.optional_index_failed",
                media_id=request.media_id,
                error=str(exc),
            )
            return {"transcript": 0, "audio": 0, "ocr": 0}

    async def _index_transcript_audio_ocr(
        self,
        *,
        request: IngestionRequested,
        canonical_path: Path,
        shots: list[DetectedShot],
        frames: list[SampledFrame],
        job_dir: Path,
    ) -> dict[str, int]:
        assert self._transcript_writer is not None
        assert self._audio_writer is not None
        assert self._ocr_writer is not None

        logger.info("ingestion.ingest.optional_indexing_start", media_id=request.media_id)
        has_audio = await self._audio_extractor.has_audio_stream(canonical_path)
        if has_audio:
            full_audio_path = await self._audio_extractor.extract_full_audio(
                canonical_path,
                job_dir / "audio",
            )
            transcript_segments = await self._transcriber.transcribe(full_audio_path)
            shot_transcripts = [
                transcript
                for transcript in align_transcript_to_shots(transcript_segments, shots)
                if transcript.text.strip()
            ]
            transcript_embeddings = await self._text_encoder.encode_texts(
                [transcript.text for transcript in shot_transcripts]
            )
            await self._transcript_writer.write_points(
                request,
                transcript_points(request, shot_transcripts, transcript_embeddings),
            )
            transcript_count = len(shot_transcripts)

            audio_clips = await self._audio_extractor.extract_shot_audio_clips(
                canonical_path,
                shots,
                job_dir / "audio_clips",
            )
            audio_embeddings = await self._audio_encoder.encode_audio_clips(audio_clips)
            await self._audio_writer.write_points(
                request,
                audio_points(request, audio_clips, audio_embeddings),
            )
            audio_count = len(audio_clips)
        else:
            await self._transcript_writer.write_points(request, [])
            await self._audio_writer.write_points(request, [])
            transcript_count = 0
            audio_count = 0

        shot_ocr_texts = [
            shot_ocr_text
            for shot_ocr_text in await self._ocr_reader.read_frames(frames)
            if shot_ocr_text.text.strip()
        ]
        ocr_embeddings = await self._text_encoder.encode_texts(
            [shot_ocr_text.text for shot_ocr_text in shot_ocr_texts]
        )
        await self._ocr_writer.write_points(
            request,
            ocr_points(request, shot_ocr_texts, ocr_embeddings),
        )
        logger.info("ingestion.ingest.optional_indexing_completed", media_id=request.media_id)
        return {"transcript": transcript_count, "audio": audio_count, "ocr": len(shot_ocr_texts)}

    async def _index_media_visual(
        self,
        request: IngestionRequested,
        frame: SampledFrame,
        metadata: ImageMetadata,
    ) -> int:
        assert self._media_visual_writer is not None
        embeddings = await self._visual_encoder.encode_frames([frame])
        if len(embeddings) != 1:
            raise ValueError(f"Expected 1 image visual embedding, got {len(embeddings)}.")
        await self._media_visual_writer.write_points(
            request,
            [image_visual_point(request, metadata, embeddings[0])],
        )
        return 1

    async def _index_media_audio(
        self,
        request: IngestionRequested,
        canonical_path: Path,
        metadata: AudioMetadata,
    ) -> int:
        assert self._media_audio_writer is not None
        clip = ShotAudioClip(
            shot=media_level_shot(request),
            path=canonical_path,
            clip_start_seconds=0.0,
            clip_end_seconds=metadata.duration_seconds or 0.0,
            clip_duration_seconds=metadata.duration_seconds or 0.0,
        )
        embeddings = await self._audio_encoder.encode_audio_clips([clip])
        if len(embeddings) != 1:
            raise ValueError(f"Expected 1 audio embedding, got {len(embeddings)}.")
        await self._media_audio_writer.write_points(
            request,
            [audio_media_point(request, metadata, embeddings[0])],
        )
        return 1

    async def _index_audio_transcript_best_effort(
        self,
        request: IngestionRequested,
        canonical_path: Path,
    ) -> int:
        assert self._media_transcript_writer is not None
        try:
            return await asyncio.wait_for(
                self._index_audio_transcript(request, canonical_path),
                timeout=self._optional_index_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "ingestion.audio.transcript_timeout",
                media_id=request.media_id,
                timeout_seconds=self._optional_index_timeout_seconds,
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingestion.audio.transcript_failed",
                media_id=request.media_id,
                error=str(exc),
            )
            return 0

    async def _index_audio_transcript(
        self,
        request: IngestionRequested,
        canonical_path: Path,
    ) -> int:
        assert self._media_transcript_writer is not None
        segments = await self._transcriber.transcribe(canonical_path)
        text = " ".join(segment.text for segment in segments if segment.text.strip()).strip()
        if not text:
            await self._media_transcript_writer.write_points(request, [])
            return 0

        embeddings = await self._text_encoder.encode_texts([text])
        if len(embeddings) != 1:
            raise ValueError(f"Expected 1 transcript embedding, got {len(embeddings)}.")
        await self._media_transcript_writer.write_points(
            request,
            [audio_transcript_point(request, text, embeddings[0], segment_count=len(segments))],
        )
        return 1

    async def _index_image_ocr_best_effort(
        self,
        request: IngestionRequested,
        frame: SampledFrame,
    ) -> int:
        assert self._media_ocr_writer is not None
        try:
            return await asyncio.wait_for(
                self._index_image_ocr(request, frame),
                timeout=self._optional_index_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "ingestion.image.ocr_timeout",
                media_id=request.media_id,
                timeout_seconds=self._optional_index_timeout_seconds,
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingestion.image.ocr_failed",
                media_id=request.media_id,
                error=str(exc),
            )
            return 0

    async def _index_image_ocr(self, request: IngestionRequested, frame: SampledFrame) -> int:
        assert self._media_ocr_writer is not None
        ocr_texts = [
            ocr_text
            for ocr_text in await self._ocr_reader.read_frames([frame])
            if ocr_text.text.strip()
        ]
        if not ocr_texts:
            await self._media_ocr_writer.write_points(request, [])
            return 0

        text = " ".join(ocr_text.text for ocr_text in ocr_texts).strip()
        embeddings = await self._text_encoder.encode_texts([text])
        if len(embeddings) != 1:
            raise ValueError(f"Expected 1 OCR embedding, got {len(embeddings)}.")
        text_block_count = sum(ocr_text.text_block_count for ocr_text in ocr_texts)
        confidences = [
            ocr_text.mean_confidence
            for ocr_text in ocr_texts
            if ocr_text.mean_confidence is not None
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        await self._media_ocr_writer.write_points(
            request,
            [
                image_ocr_point(
                    request,
                    text,
                    embeddings[0],
                    text_block_count=text_block_count,
                    mean_confidence=mean_confidence,
                )
            ],
        )
        return 1


def build_embedding_writer(
    *,
    qdrant: QdrantClient | None,
    writer: ShotEmbeddingIndexWriter | None,
    collection_name: str,
    embedding_dim: int,
) -> ShotEmbeddingIndexWriter | None:
    if writer is not None:
        return writer
    if qdrant is None:
        return None
    return QdrantShotEmbeddingWriter(
        qdrant,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
    )


def canonical_filename(object_key: str) -> str:
    return Path(object_key).name or "canonical.mp4"


@contextmanager
def temporary_job_dir(work_dir: Path) -> Iterator[Path]:
    job_dir = work_dir / f"job-{uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield job_dir
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def merge_metadata(
    *,
    request_metadata: VideoMetadata,
    probed_metadata: VideoMetadata,
) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=probed_metadata.duration_seconds or request_metadata.duration_seconds,
        width=probed_metadata.width or request_metadata.width,
        height=probed_metadata.height or request_metadata.height,
        frame_rate=probed_metadata.frame_rate or request_metadata.frame_rate,
        codec=probed_metadata.codec or request_metadata.codec,
    )


def merge_audio_metadata(
    *,
    request_metadata: AudioMetadata,
    probed_metadata: AudioMetadata,
) -> AudioMetadata:
    return AudioMetadata(
        duration_seconds=probed_metadata.duration_seconds or request_metadata.duration_seconds,
        codec=probed_metadata.codec or request_metadata.codec,
    )


def merge_image_metadata(
    *,
    request_metadata: ImageMetadata,
    probed_metadata: ImageMetadata,
) -> ImageMetadata:
    return ImageMetadata(
        width=probed_metadata.width or request_metadata.width,
        height=probed_metadata.height or request_metadata.height,
    )


def media_level_shot(request: IngestionRequested) -> DetectedShot:
    return DetectedShot(
        shot_id=f"{request.media_id}:{request.kind.value.lower()}",
        media_id=request.media_id,
        shot_index=0,
        start_seconds=0.0,
        end_seconds=request.duration_seconds or 0.0,
        duration_seconds=request.duration_seconds or 0.0,
    )


def media_image_frame(request: IngestionRequested, image_path: Path) -> SampledFrame:
    return SampledFrame(
        shot=media_level_shot(request),
        timestamp_seconds=0.0,
        path=image_path,
    )
