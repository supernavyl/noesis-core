"""Ingestion base class.

Every source implements ``fetch_new()`` to yield raw items. The pipeline then chunks, embeds,
and writes them to the vector store. Dedup is handled by content-hashed point IDs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class IngestionItem:
    source: str
    source_id: str
    url: str | None
    title: str
    text: str
    published_at: str | None = None
    authors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    source: str
    fetched: int = 0
    written: int = 0
    skipped: int = 0
    errors: int = 0


class IngestionSource(ABC):
    name: str

    @abstractmethod
    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        """Yield items not yet ingested."""
        if False:  # type-checker hint that this is an async generator
            yield  # pragma: no cover
