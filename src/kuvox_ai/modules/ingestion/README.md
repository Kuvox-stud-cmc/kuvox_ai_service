# `ingestion`

The ingestion module turns an optimized canonical video into graph-backed shot data and per-shot multimodal vectors after media storage optimization has completed.

Current MVP 3 behavior:

- Consumes `ingestion.requested` from the `kuvox.events` direct exchange.
- Accepts library-scoped events with `mediaId`, `ownerId`, `ownerKind`, `kind`, canonical object, optional proxy/thumbnail objects, and metadata. It does not accept `projectId`.
- Downloads only the canonical video object.
- Runs FFprobe metadata and PySceneDetect shot detection.
- Falls back to one full-length shot when no scene boundaries are detected.
- Writes `Video` and `Shot` nodes plus `BELONGS_TO` and `NEXT` relationships to Kuzu.
- Samples one midpoint JPEG frame per detected shot with FFmpeg.
- Encodes sampled frames with OpenCLIP using normalized visual embeddings.
- Ensures the Qdrant `shots_visual` collection exists with cosine distance and the configured embedding dimension.
- Deletes existing Qdrant visual points for the same `mediaId`, then upserts one point per shot keyed by deterministic `shotId`.
- Extracts video audio when present, transcribes with Faster Whisper, maps transcript segments to overlapping shots, embeds shot transcript text with SentenceTransformers, and writes `shots_transcript`.
- Extracts per-shot audio clips when audio is present, embeds them with MS-CLAP, and writes `shots_audio`.
- Runs EasyOCR on the sampled midpoint frames, embeds non-empty OCR text with SentenceTransformers, and writes `shots_ocr`.
- Deletes existing transcript/audio points for no-audio videos and continues successfully.
- Publishes `ingestion.completed` only after graph writes and all configured vector indexes succeed; model, FFmpeg, OCR, encoder, or Qdrant failures are retried by the worker before `ingestion.failed` is published on retry exhaustion.

Visual point payloads include `mediaId`, `ownerId`, `ownerKind`, `shotId`, `shotIndex`, `startSeconds`, `endSeconds`, `durationSeconds`, and `frameTimestampSeconds`.

Transcript, audio, and OCR point payloads include the same media/owner/shot timing fields plus modality-specific text, confidence, segment, or clip metadata.

Future slices should add one capability at a time. Retrieval over `shots_visual`, `shots_transcript`, `shots_audio`, and `shots_ocr`, scenes, retries, DLQs, outbox, and graph algorithms are intentionally not part of MVP 3.

HTTP callers should not invoke ingestion directly. The worker entry point is `kuvox_ai/workers/ingestion_worker.py`.
