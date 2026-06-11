"""Audit log — every reasoning trace + every weight update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg

from noesis.core.config import settings


@dataclass
class AuditEvent:
    event_type: str
    actor: str
    summary: str
    payload: dict[str, Any] | None = None
    principle_ref: str | None = None


class AuditLog:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings().postgres_dsn

    async def write(self, event: AuditEvent) -> int:
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO audit (event_type, actor, summary, payload, constitution_principle_ref)
                       VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                    (
                        event.event_type,
                        event.actor,
                        event.summary,
                        event.payload,
                        event.principle_ref,
                    ),
                )
                row = await cur.fetchone()
                await conn.commit()
                return int(row[0]) if row else -1
