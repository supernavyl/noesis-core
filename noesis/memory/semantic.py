"""Semantic memory — facts and concepts extracted from corpus, stored in Qdrant.

Backed by the same Qdrant collection as the knowledge corpus, but tagged with
``source=semantic`` for facts distilled from interactions, vs ``source=arxiv``,
``source=blog`` etc. for ingested raw content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from noesis.knowledge.vector_store import VectorStore


@dataclass
class SemanticFact:
    statement: str
    confidence: float
    provenance: list[str]  # source chunk ids or URLs
    embedding: list[float] | None = None
    meta: dict[str, Any] | None = None


class SemanticStore:
    """Thin wrapper on the shared Qdrant collection for distilled facts."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vs = vector_store or VectorStore()

    async def write(self, fact: SemanticFact) -> str:
        return await self.vs.upsert_text(
            text=fact.statement,
            embedding=fact.embedding,
            payload={
                "type": "semantic_fact",
                "confidence": fact.confidence,
                "provenance": fact.provenance,
                **(fact.meta or {}),
            },
        )

    async def query(self, q: str, k: int = 8) -> list[dict[str, Any]]:
        return await self.vs.search(q, k=k, filter_={"type": "semantic_fact"})
