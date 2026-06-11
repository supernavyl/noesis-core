"""Ingestion pipeline runner — orchestrates all sources, chunks, embeds, writes.

This is the daily / per-source entry point. Wires the source-specific fetchers to chunking
and the vector store.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import psycopg
from loguru import logger
from qdrant_client.models import PointStruct, SparseVector

from noesis.core.config import settings
from noesis.knowledge.chunking import (
    build_parent_chunks,
    fixed_chunks,
    semantic_chunks,
    split_sentences,
)
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource
from noesis.knowledge.vector_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    VectorStore,
    _id_for_text,
)

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]
SparseFn = Callable[[str], Awaitable[dict[int, float]]]


@dataclass
class PipelineStats:
    source: str
    fetched: int = 0
    chunks_written: int = 0
    skipped_duplicate: int = 0
    errors: int = 0
    error_samples: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        store: VectorStore | None = None,
        embed_fn: EmbedFn | None = None,
        sparse_fn: SparseFn | None = None,
        semantic_chunking: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.store = store or VectorStore()
        self.embed_fn = embed_fn
        self.sparse_fn = sparse_fn
        self.semantic_chunking = semantic_chunking
        self.chunk_size = chunk_size

    @classmethod
    def default(cls, semantic_chunking: bool = True) -> IngestionPipeline:
        """Wire production stack: TEI embedder + SPLADE sparse encoder."""
        from noesis.clients.embedder import EmbedderClient
        from noesis.clients.sparse import SpladeEncoder

        embedder = EmbedderClient()
        sparse = SpladeEncoder()

        async def embed_fn(texts: list[str]) -> list[list[float]]:
            return await embedder.embed(texts)

        async def sparse_fn(text: str) -> dict[int, float]:
            return await sparse.encode(text)

        return cls(
            store=VectorStore(),
            embed_fn=embed_fn,
            sparse_fn=sparse_fn,
            semantic_chunking=semantic_chunking,
        )

    async def run_source(self, source: IngestionSource) -> PipelineStats:
        await self.store.ensure_collection()
        stats = PipelineStats(source=source.name)

        async for item in source.fetch_new():
            stats.fetched += 1
            try:
                await self._process_item(item)
                stats.chunks_written += 1
                await self._record_ingested(item)
            except DuplicateError:
                stats.skipped_duplicate += 1
            except Exception as exc:
                stats.errors += 1
                if len(stats.error_samples) < 5:
                    stats.error_samples.append(f"{type(exc).__name__}: {exc}")
                logger.exception("Ingestion error for {}/{}", source.name, item.source_id)

        logger.info(
            "source={} fetched={} written={} dup={} err={}",
            source.name,
            stats.fetched,
            stats.chunks_written,
            stats.skipped_duplicate,
            stats.errors,
        )
        return stats

    async def _process_item(self, item: IngestionItem) -> None:
        if await self._already_ingested(item):
            raise DuplicateError(item.source_id)

        # Chunk
        if self.semantic_chunking and self.embed_fn:
            sentences = split_sentences(item.text)
            if not sentences:
                return
            sent_embeds = await self.embed_fn(sentences)
            child_chunks = semantic_chunks(sentences, sent_embeds)
        else:
            child_chunks = fixed_chunks(item.text, size=self.chunk_size, overlap=self.chunk_overlap)

        parent_chunks = build_parent_chunks(child_chunks)

        # Embed + sparse-encode
        chunk_texts = [c.text for c in child_chunks]
        if not chunk_texts:
            return

        dense = await self.embed_fn(chunk_texts) if self.embed_fn else [None] * len(chunk_texts)
        sparse_vecs: list[dict[int, float] | None] = (
            await asyncio.gather(*[self.sparse_fn(t) for t in chunk_texts])
            if self.sparse_fn
            else [None] * len(chunk_texts)
        )

        # Build points
        base_payload = {
            "source": item.source,
            "source_id": item.source_id,
            "url": item.url,
            "title": item.title,
            "authors": item.authors,
            "tags": item.tags,
            "published_at": item.published_at,
        }
        points = []
        for i, ((chunk, emb), sp) in enumerate(zip(zip(child_chunks, dense), sparse_vecs)):
            vector: dict = {}
            if emb is not None:
                vector[DENSE_VECTOR_NAME] = emb
            if sp is not None:
                vector[SPARSE_VECTOR_NAME] = SparseVector(
                    indices=list(sp.keys()), values=list(sp.values())
                )
            points.append(
                PointStruct(
                    id=_id_for_text(f"{item.source_id}::{i}::{chunk.text[:64]}"),
                    vector=vector,
                    payload={
                        **base_payload,
                        "text": chunk.text,
                        "chunk_idx": i,
                        "parent_text": parent_chunks[i // 4].text if parent_chunks else None,
                    },
                )
            )
        await self.store.upsert_many(points)

    async def _already_ingested(self, item: IngestionItem) -> bool:
        async with await psycopg.AsyncConnection.connect(settings().postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM ingested WHERE source=%s AND source_id=%s",
                    (item.source, item.source_id),
                )
                row = await cur.fetchone()
                return row is not None

    async def _record_ingested(self, item: IngestionItem) -> None:
        async with await psycopg.AsyncConnection.connect(settings().postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO ingested (source, source_id, url, title, bytes)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (source, source_id) DO NOTHING""",
                    (item.source, item.source_id, item.url, item.title, len(item.text)),
                )
                await conn.commit()


class DuplicateError(RuntimeError):
    pass
