"""Procedural memory — compiled skills, prompt templates, tool patterns.

Stored in SQLite with FTS5 for fast keyword lookup. Used by the orchestrator
to recall "how do I do X" patterns the model has learned.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noesis.core.config import settings


@dataclass
class Procedure:
    name: str
    description: str
    body: str
    tags: list[str]
    use_count: int = 0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    name, description, body, tags, content='procedures', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
    INSERT INTO procedures_fts(rowid, name, description, body, tags)
    VALUES (new.id, new.name, new.description, new.body, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, name, description, body, tags)
    VALUES ('delete', old.id, old.name, old.description, old.body, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS procedures_au AFTER UPDATE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, name, description, body, tags)
    VALUES ('delete', old.id, old.name, old.description, old.body, old.tags);
    INSERT INTO procedures_fts(rowid, name, description, body, tags)
    VALUES (new.id, new.name, new.description, new.body, new.tags);
END;
"""


class ProceduralStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (settings().noesis_data_dir / "procedural.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def write(self, proc: Procedure) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO procedures (name, description, body, tags)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                     description=excluded.description,
                     body=excluded.body,
                     tags=excluded.tags
                   RETURNING id""",
                (proc.name, proc.description, proc.body, ",".join(proc.tags)),
            )
            row = cur.fetchone()
            return int(row[0]) if row else -1

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                """SELECT p.id, p.name, p.description, p.body, p.tags, p.use_count
                   FROM procedures_fts f
                   JOIN procedures p ON p.id = f.rowid
                   WHERE procedures_fts MATCH ?
                   ORDER BY rank, p.use_count DESC
                   LIMIT ?""",
                (query, limit),
            )
            return [dict(r) for r in cur.fetchall()]
