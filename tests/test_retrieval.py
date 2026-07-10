"""HybridRetriever end-to-end test with stub backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from noesis.knowledge.retrieval import HybridRetriever, Retrieved


@dataclass
class StubVectorStore:
    rows: list[dict[str, Any]]

    async def search(self, *_a: Any, **_kw: Any) -> list[dict[str, Any]]:
        return self.rows


class StubEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class StubSparse:
    async def encode(self, _text: str) -> dict[int, float]:
        return {1: 1.0}


class StubReranker:
    async def rerank(self, _query: str, candidates: list[str]) -> list[float]:
        # Reverse-sort by length so the test asserts ordering really changed.
        return [float(len(c)) for c in candidates]


def test_hybrid_retriever_reorders_by_reranker() -> None:
    rows = [
        {"id": "a", "score": 0.5, "payload": {"text": "short"}},
        {"id": "b", "score": 0.4, "payload": {"text": "this is a much longer chunk of text"}},
        {"id": "c", "score": 0.3, "payload": {"text": "medium length"}},
    ]
    retriever = HybridRetriever(
        store=StubVectorStore(rows),  # type: ignore[arg-type]
        embedder=StubEmbedder(),
        sparse=StubSparse(),
        reranker=StubReranker(),
    )

    async def _run() -> list[Retrieved]:
        return await retriever.search("anything", k_candidates=10, k_final=3)

    out = asyncio.run(_run())
    assert len(out) == 3
    # Longest text should now be first since reranker scored by length.
    assert "longer chunk" in out[0].text


def test_hybrid_retriever_no_reranker_preserves_order() -> None:
    rows = [
        {"id": "a", "score": 0.9, "payload": {"text": "first"}},
        {"id": "b", "score": 0.5, "payload": {"text": "second"}},
    ]
    retriever = HybridRetriever(
        store=StubVectorStore(rows),  # type: ignore[arg-type]
        embedder=StubEmbedder(),
        sparse=StubSparse(),
        reranker=None,
    )
    out = asyncio.run(retriever.search("q", k_candidates=10, k_final=5))
    assert out[0].text == "first"
    assert out[1].text == "second"


def test_hybrid_retriever_uses_hyde_for_dense_when_enabled() -> None:
    """When use_hyde=True, the embedder gets called on the expanded text, not the raw query."""

    rows = [{"id": "x", "score": 0.5, "payload": {"text": "doc", "source": "arxiv"}}]

    class _StubStore:
        async def search(self, *_a, **_kw):
            return rows

    class _RecordingEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 2.0, 3.0] for _ in texts]

    class _StubSparse:
        async def encode(self, _text):
            return {}

    class _StubHyde:
        async def expand(self, q):
            return f"EXPANDED: {q}"

    embedder = _RecordingEmbedder()
    retriever = HybridRetriever(
        store=_StubStore(),
        embedder=embedder,
        sparse=_StubSparse(),
        reranker=None,
        hyde=_StubHyde(),
        priority_weights={"arxiv": 0.5},
        priority_blend=0.0,
    )

    import asyncio

    hits = asyncio.run(retriever.search("flash attention?", use_hyde=True))
    assert len(hits) == 1
    assert embedder.calls == [["EXPANDED: flash attention?"]]


def test_hybrid_retriever_skips_hyde_when_disabled() -> None:
    rows = [{"id": "x", "score": 0.5, "payload": {"text": "doc", "source": "arxiv"}}]

    class _StubStore:
        async def search(self, *_a, **_kw):
            return rows

    class _RecordingEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed(self, texts):
            self.calls.append(list(texts))
            return [[1.0] for _ in texts]

    class _StubSparse:
        async def encode(self, _text):
            return {}

    class _StubHyde:
        async def expand(self, q):
            raise AssertionError("HyDE should not be called when use_hyde=False")

    embedder = _RecordingEmbedder()
    retriever = HybridRetriever(
        store=_StubStore(),
        embedder=embedder,
        sparse=_StubSparse(),
        reranker=None,
        hyde=_StubHyde(),
    )

    import asyncio

    asyncio.run(retriever.search("raw query", use_hyde=False))
    assert embedder.calls == [["raw query"]]
