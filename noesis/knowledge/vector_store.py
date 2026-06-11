"""Qdrant vector store with hybrid (dense + sparse) search.

Single collection ``noesis_corpus`` with both dense (Qwen3-Embedding-8B, 4096-dim) and sparse
(SPLADE-v3) vectors. Server-side RRF fusion via the Qdrant Query API.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from noesis.core.config import settings

COLLECTION = "noesis_corpus"
DENSE_DIM = 4096  # Qwen3-Embedding-8B
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class VectorStore:
    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        s = settings()
        self.client = AsyncQdrantClient(
            url=url or s.qdrant_url, api_key=api_key or s.qdrant_api_key
        )

    async def ensure_collection(self) -> None:
        existing = await self.client.get_collections()
        if any(c.name == COLLECTION for c in existing.collections):
            return
        await self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )

    async def upsert_text(
        self,
        text: str,
        embedding: list[float] | None,
        sparse: dict[int, float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        point_id = _id_for_text(text)
        vectors: dict[str, Any] = {}
        if embedding is not None:
            vectors[DENSE_VECTOR_NAME] = embedding
        if sparse is not None:
            vectors[SPARSE_VECTOR_NAME] = SparseVector(
                indices=list(sparse.keys()), values=list(sparse.values())
            )
        await self.client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload={"text": text, **(payload or {})},
                )
            ],
        )
        return point_id

    async def upsert_many(self, points: list[PointStruct]) -> None:
        # Chunk to keep request size sane
        for i in range(0, len(points), 256):
            await self.client.upsert(collection_name=COLLECTION, points=points[i : i + 256])

    async def search(
        self,
        query: str,
        k: int = 16,
        embedding: list[float] | None = None,
        sparse: dict[int, float] | None = None,
        filter_: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search via Qdrant Query API. RRF-fuses dense + sparse results."""
        prefetch: list[Any] = []
        if embedding is not None:
            prefetch.append({"query": embedding, "using": DENSE_VECTOR_NAME, "limit": k * 3})
        if sparse is not None:
            prefetch.append(
                {
                    "query": SparseVector(
                        indices=list(sparse.keys()), values=list(sparse.values())
                    ),
                    "using": SPARSE_VECTOR_NAME,
                    "limit": k * 3,
                }
            )

        qdrant_filter = _build_filter(filter_)
        result = await self.client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch if prefetch else None,
            query={"fusion": "rrf"} if len(prefetch) > 1 else None,
            limit=k,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        return [{"id": p.id, "score": p.score, "payload": p.payload} for p in result.points]

    async def count(self) -> int:
        res = await self.client.count(collection_name=COLLECTION, exact=True)
        return res.count


def _id_for_text(text: str) -> str:
    # Deterministic UUID-shaped id from content hash — enables natural dedup.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _build_filter(spec: dict[str, Any] | None) -> Filter | None:
    if not spec:
        return None
    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in spec.items()])


async def quick_setup() -> None:
    """Helper for scripts — initialize the collection."""
    store = VectorStore()
    await store.ensure_collection()


if __name__ == "__main__":
    asyncio.run(quick_setup())
