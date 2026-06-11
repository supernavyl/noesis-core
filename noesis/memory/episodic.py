"""Episodic memory — every interaction, persisted in Postgres + pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from noesis.core.config import settings


@dataclass
class EpisodicEvent:
    session_id: str
    role: str
    content: str
    embedding: list[float] | None = None
    tokens: int | None = None
    model: str | None = None
    reasoning_depth: int | None = None
    meta: dict[str, Any] | None = None


class EpisodicStore:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings().postgres_dsn

    async def write(self, event: EpisodicEvent) -> int:
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO episodic
                        (session_id, role, content, embedding, tokens, model, reasoning_depth, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        event.session_id,
                        event.role,
                        event.content,
                        event.embedding,
                        event.tokens,
                        event.model,
                        event.reasoning_depth,
                        event.meta,
                    ),
                )
                row = await cur.fetchone()
                await conn.commit()
                return row[0] if row else -1

    async def recent_session(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM episodic WHERE session_id = %s ORDER BY ts DESC LIMIT %s",
                    (session_id, limit),
                )
                return list(await cur.fetchall())

    async def recent_window(self, days: int = 7) -> list[dict[str, Any]]:
        """Used by dream cycle NREM phase to replay recent episodes for consolidation."""
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM episodic WHERE ts > NOW() - INTERVAL '%s days' ORDER BY ts ASC",
                    (days,),
                )
                return list(await cur.fetchall())
