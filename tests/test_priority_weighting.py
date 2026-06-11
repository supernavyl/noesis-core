"""Priority weighting in HybridRetriever."""

from __future__ import annotations

import asyncio
from typing import Any

from noesis.knowledge.retrieval import HybridRetriever


class _StubStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def search(self, *_a: Any, **_kw: Any) -> list[dict[str, Any]]:
        return self.rows


class _StubEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _StubSparse:
    async def encode(self, _text: str) -> dict[int, float]:
        return {}


class _StubReranker:
    """Returns identical rerank scores so priority weights drive ordering."""

    async def rerank(self, _q: str, candidates: list[str]) -> list[float]:
        return [0.5] * len(candidates)


def test_priority_reweights_when_rerank_scores_tied() -> None:
    rows = [
        {"id": "1", "score": 0.5, "payload": {"text": "from reddit", "source": "reddit"}},
        {"id": "2", "score": 0.5, "payload": {"text": "from arxiv", "source": "arxiv"}},
        {"id": "3", "score": 0.5, "payload": {"text": "from books", "source": "books"}},
    ]
    retriever = HybridRetriever(
        store=_StubStore(rows),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        sparse=_StubSparse(),
        reranker=_StubReranker(),
        priority_weights={"books": 1.0, "arxiv": 0.8, "reddit": 0.3},
        priority_blend=0.5,
    )
    out = asyncio.run(retriever.search("q", k_candidates=10, k_final=3))
    # Books outrank arxiv outrank reddit
    sources = [r.payload["source"] for r in out]
    assert sources == ["books", "arxiv", "reddit"]


def test_priority_blend_zero_falls_back_to_pure_rerank() -> None:
    rows = [
        {"id": "1", "score": 0.5, "payload": {"text": "low-prio", "source": "reddit"}},
        {"id": "2", "score": 0.5, "payload": {"text": "high-prio", "source": "books"}},
    ]

    class HighFirst:
        async def rerank(self, _q: str, candidates: list[str]) -> list[float]:
            return [9.0, 0.1]  # first candidate wins on rerank

    retriever = HybridRetriever(
        store=_StubStore(rows),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        sparse=_StubSparse(),
        reranker=HighFirst(),
        priority_weights={"books": 1.0, "reddit": 0.0},
        priority_blend=0.0,
    )
    out = asyncio.run(retriever.search("q", k_candidates=10, k_final=2))
    # With blend=0, the rerank-winning reddit chunk stays first
    assert out[0].payload["source"] == "reddit"


def test_default_priority_loads_from_yaml() -> None:
    retriever = HybridRetriever(
        store=_StubStore([]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        sparse=_StubSparse(),
    )
    # The ingestion.yaml ships with weights for arxiv, books, etc.
    assert retriever.priority_weights.get("arxiv", 0.0) > 0
    assert retriever.priority_weights.get("books", 0.0) > 0
