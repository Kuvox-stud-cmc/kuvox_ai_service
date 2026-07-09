"""Kuzu graph writes for ingestion MVP 1."""

from __future__ import annotations

from kuvox_ai.infrastructure import KuzuClient
from kuvox_ai.modules.ingestion.models import (
    AudioMetadata,
    DetectedShot,
    ImageMetadata,
    IngestionRequested,
    VideoMetadata,
)


class KuzuIngestionWriter:
    def __init__(self, kuzu: KuzuClient) -> None:
        self._kuzu = kuzu

    async def ensure_schema(self) -> None:
        await self._kuzu.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Video(
                media_id STRING,
                owner_id STRING,
                owner_kind STRING,
                kind STRING,
                canonical_object_key STRING,
                duration_seconds DOUBLE,
                width INT64,
                height INT64,
                frame_rate DOUBLE,
                codec STRING,
                PRIMARY KEY(media_id)
            )
            """
        )
        await self._kuzu.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Shot(
                shot_id STRING,
                media_id STRING,
                shot_index INT64,
                start_seconds DOUBLE,
                end_seconds DOUBLE,
                duration_seconds DOUBLE,
                PRIMARY KEY(shot_id)
            )
            """
        )
        await self._kuzu.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Audio(
                media_id STRING,
                owner_id STRING,
                owner_kind STRING,
                kind STRING,
                canonical_object_key STRING,
                duration_seconds DOUBLE,
                codec STRING,
                PRIMARY KEY(media_id)
            )
            """
        )
        await self._kuzu.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Image(
                media_id STRING,
                owner_id STRING,
                owner_kind STRING,
                kind STRING,
                canonical_object_key STRING,
                width INT64,
                height INT64,
                PRIMARY KEY(media_id)
            )
            """
        )
        await self._kuzu.execute("CREATE REL TABLE IF NOT EXISTS BELONGS_TO(FROM Shot TO Video)")
        await self._kuzu.execute("CREATE REL TABLE IF NOT EXISTS NEXT(FROM Shot TO Shot)")

    async def write_video_with_shots(
        self,
        request: IngestionRequested,
        metadata: VideoMetadata,
        shots: list[DetectedShot],
    ) -> None:
        await self.ensure_schema()
        await self._delete_existing_shots(request.media_id)
        await self._upsert_video(request, metadata)

        previous: DetectedShot | None = None
        for shot in shots:
            await self._create_shot(shot)
            await self._kuzu.execute(
                """
                MATCH (s:Shot {shot_id: $shot_id}), (v:Video {media_id: $media_id})
                CREATE (s)-[:BELONGS_TO]->(v)
                """,
                {"shot_id": shot.shot_id, "media_id": request.media_id},
            )
            if previous is not None:
                await self._kuzu.execute(
                    """
                    MATCH (a:Shot {shot_id: $previous_shot_id}), (b:Shot {shot_id: $shot_id})
                    CREATE (a)-[:NEXT]->(b)
                    """,
                    {"previous_shot_id": previous.shot_id, "shot_id": shot.shot_id},
                )
            previous = shot

    async def write_audio(self, request: IngestionRequested, metadata: AudioMetadata) -> None:
        await self.ensure_schema()
        await self._kuzu.execute(
            """
            MERGE (a:Audio {media_id: $media_id})
            SET a.owner_id = $owner_id,
                a.owner_kind = $owner_kind,
                a.kind = $kind,
                a.canonical_object_key = $canonical_object_key,
                a.duration_seconds = $duration_seconds,
                a.codec = $codec
            """,
            {
                "media_id": request.media_id,
                "owner_id": request.owner_id,
                "owner_kind": request.owner_kind.value,
                "kind": request.kind.value,
                "canonical_object_key": request.canonical.object_key,
                "duration_seconds": metadata.duration_seconds,
                "codec": metadata.codec,
            },
        )

    async def write_image(self, request: IngestionRequested, metadata: ImageMetadata) -> None:
        await self.ensure_schema()
        await self._kuzu.execute(
            """
            MERGE (i:Image {media_id: $media_id})
            SET i.owner_id = $owner_id,
                i.owner_kind = $owner_kind,
                i.kind = $kind,
                i.canonical_object_key = $canonical_object_key,
                i.width = $width,
                i.height = $height
            """,
            {
                "media_id": request.media_id,
                "owner_id": request.owner_id,
                "owner_kind": request.owner_kind.value,
                "kind": request.kind.value,
                "canonical_object_key": request.canonical.object_key,
                "width": metadata.width,
                "height": metadata.height,
            },
        )

    async def _delete_existing_shots(self, media_id: str) -> None:
        await self._kuzu.execute(
            "MATCH (s:Shot)-[r:BELONGS_TO]->(v:Video {media_id: $media_id}) DELETE r",
            {"media_id": media_id},
        )
        await self._kuzu.execute(
            "MATCH (a:Shot)-[r:NEXT]->(b:Shot) WHERE a.media_id = $media_id DELETE r",
            {"media_id": media_id},
        )
        await self._kuzu.execute(
            "MATCH (s:Shot {media_id: $media_id}) DELETE s",
            {"media_id": media_id},
        )

    async def _upsert_video(self, request: IngestionRequested, metadata: VideoMetadata) -> None:
        await self._kuzu.execute(
            """
            MERGE (v:Video {media_id: $media_id})
            SET v.owner_id = $owner_id,
                v.owner_kind = $owner_kind,
                v.kind = $kind,
                v.canonical_object_key = $canonical_object_key,
                v.duration_seconds = $duration_seconds,
                v.width = $width,
                v.height = $height,
                v.frame_rate = $frame_rate,
                v.codec = $codec
            """,
            {
                "media_id": request.media_id,
                "owner_id": request.owner_id,
                "owner_kind": request.owner_kind.value,
                "kind": request.kind.value,
                "canonical_object_key": request.canonical.object_key,
                "duration_seconds": metadata.duration_seconds,
                "width": metadata.width,
                "height": metadata.height,
                "frame_rate": metadata.frame_rate,
                "codec": metadata.codec,
            },
        )

    async def _create_shot(self, shot: DetectedShot) -> None:
        await self._kuzu.execute(
            """
            CREATE (:Shot {
                shot_id: $shot_id,
                media_id: $media_id,
                shot_index: $shot_index,
                start_seconds: $start_seconds,
                end_seconds: $end_seconds,
                duration_seconds: $duration_seconds
            })
            """,
            shot.model_dump(),
        )
