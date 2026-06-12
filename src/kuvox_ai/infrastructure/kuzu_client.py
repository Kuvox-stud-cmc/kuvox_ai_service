"""Async wrapper around the embedded Kuzu graph database.

Kuzu's Python binding is synchronous; we offload calls onto a thread to keep
the FastAPI event loop responsive.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import kuzu

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

logger = get_logger(__name__)


class KuzuClient:
    """Connection holder for the local Kuzu database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> KuzuClient:
        return cls(db_path=settings.kuzu_db_path)

    async def connect(self) -> None:
        # Kuzu stores a database as a single file; create only the parent
        # directory and let Kuzu create the db file at ``_db_path``.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("kuzu.connecting", path=str(self._db_path))

        def _open() -> tuple[kuzu.Database, kuzu.Connection]:
            db = kuzu.Database(str(self._db_path))
            return db, kuzu.Connection(db)

        self._db, self._conn = await asyncio.to_thread(_open)
        logger.info("kuzu.connected")

    async def close(self) -> None:
        if self._conn is None and self._db is None:
            return
        logger.info("kuzu.closing")
        # Kuzu has no explicit close; releasing references is sufficient.
        self._conn = None
        self._db = None
        logger.info("kuzu.closed")

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query and return the raw QueryResult.

        TODO: wrap results in typed records once the schema is finalized.
        """
        if self._conn is None:
            raise RuntimeError("KuzuClient is not connected")
        conn = self._conn
        return await asyncio.to_thread(conn.execute, query, params or {})

    async def health_check(self) -> bool:
        try:
            if self._conn is None:
                return False
            await self.execute("RETURN 1")
            return True
        except Exception as exc:  # noqa: BLE001 — health check must never raise
            logger.warning("kuzu.health_check_failed", error=str(exc))
            return False
