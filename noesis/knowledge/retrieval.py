"""Hybrid retrieval — dense + sparse + rerank, with parent-chunk expansion.

Pipeline:
1. Embed query (Qwen3-Embedding-8B)
2. SPLADE-v3 sparse vector for query
3. Qdrant hybrid search (RRF-fused) → top-K candidates
4. Cross-encoder rerank (BGE-Reranker-v2-Gemma) → top-N
5. Source-priority reweighting (books > arxiv > rss > reddit > …)
6. Parent-chunk expansion → return larger context windows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from noesis.core.config import ingestion_config
from noesis.knowledge.vector_store import VectorStore


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SparseEncoder(Protocol):
    async def encode(self, text: str) -> dict[int, float]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[str]) -> list[float]: ...


@dataclass
class Retrieved:
    text: str
    score: float
    payload: dict[str, Any]


def _load_priority_weights() -> dict[str, float]:
    try:
        cfg = ingestion_config()
        return dict(cfg.get("priority_weights", {}))
    except Exception:
        return {}


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        sparse: SparseEncoder | None = None,
        reranker: Reranker | None = None,
        hyde: Any | None = None,
        priority_weights: dict[str, float] | None = None,
        priority_blend: float = 0.25,
    ) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder
        self.sparse = sparse
        self.reranker = reranker
        self.hyde = hyde
        self.priority_weights = (
            priority_weights if priority_weights is not None else _load_priority_weights()
        )
        # Blend factor: final = (1 - blend) * rerank_score + blend * priority_weight
        self.priority_blend = max(0.0, min(1.0, priority_blend))

    @classmethod
    def default(cls) -> HybridRetriever:
        """Wire up the default production stack: TEI embedder/reranker + SPLADE sparse."""
        from noesis.clients.embedder import EmbedderClient
        from noesis.clients.llm import VLLMClient
        from noesis.clients.reranker import RerankerClient
        from noesis.clients.sparse import SpladeEncoder
        from noesis.reasoning.hyde import HypotheticalExpander

        return cls(
            store=VectorStore(),
            embedder=EmbedderClient(),
            sparse=SpladeEncoder(),
            reranker=RerankerClient(),
            hyde=HypotheticalExpander(llm=VLLMClient(served_model="reasoner.small")),
        )

    def _priority(self, payload: dict[str, Any]) -> float:
        src = payload.get("source")
        if not src:
            return 0.5
        return float(self.priority_weights.get(src, 0.5))

    async def search(
        self,
        query: str,
        k_candidates: int = 64,
        k_final: int = 8,
        filter_: dict[str, Any] | None = None,
        use_hyde: bool = False,
    ) -> list[Retrieved]:
        dense_query = query
        if use_hyde and self.hyde is not None:
            dense_query = await self.hyde.expand(query)

        emb = (await self.embedder.embed([dense_query]))[0] if self.embedder else None
        sp = await self.sparse.encode(query) if self.sparse else None

        candidates = await self.store.search(
            query=query, k=k_candidates, embedding=emb, sparse=sp, filter_=filter_
        )
        if not candidates:
            return []

        texts = [c["payload"].get("text", "") for c in candidates]

        # Stage 1: rerank scores (or fall back to retrieval scores)
        if self.reranker:
            rerank_scores = await self.reranker.rerank(query, texts)
        else:
            rerank_scores = [c["score"] for c in candidates]

        # Stage 2: blend with source priority weights
        # Normalize rerank scores into [0, 1] window for blending stability
        max_rs = max(rerank_scores) if rerank_scores else 1.0
        min_rs = min(rerank_scores) if rerank_scores else 0.0
        span = max_rs - min_rs if max_rs > min_rs else 1.0

        blended: list[tuple[dict[str, Any], float]] = []
        for cand, rs in zip(candidates, rerank_scores, strict=False):
            norm = (rs - min_rs) / span
            prio = self._priority(cand["payload"])
            final = (1.0 - self.priority_blend) * norm + self.priority_blend * prio
            blended.append((cand, final))

        blended.sort(key=lambda x: -x[1])
        top = blended[:k_final]

        return [
            Retrieved(
                text=c["payload"].get("text", ""),
                score=final_score,
                payload=c["payload"],
            )
            for c, final_score in top
        ]
