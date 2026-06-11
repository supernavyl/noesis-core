"""Embedder + reranker HTTP clients — mocked transport."""

from __future__ import annotations

import asyncio

import httpx

from noesis.clients.embedder import EmbedderClient
from noesis.clients.reranker import RerankerClient


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_embedder_returns_vectors_ollama_backend() -> None:
    """Default backend = Ollama: POST /api/embed → {embeddings: [...]}."""
    import orjson

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = orjson.loads(request.content) if request.content else {}
        n = len(body.get("input", [])) or 1
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in range(n)]})

    async def _run() -> None:
        client = EmbedderClient(base_url="http://stub", backend="ollama")
        client._http = httpx.AsyncClient(transport=_mock_transport(handler))
        try:
            vecs = await client.embed(["one", "two"])
            assert len(vecs) == 2
            assert all(len(v) == 3 for v in vecs)
        finally:
            await client._http.aclose()

    asyncio.run(_run())


def test_embedder_returns_vectors_tei_backend() -> None:
    """TEI backend: POST /embed → [[...]] flat list."""
    import orjson

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = orjson.loads(request.content) if request.content else {}
        n = len(body.get("inputs", [])) or 1
        return httpx.Response(200, json=[[0.1, 0.2, 0.3] for _ in range(n)])

    async def _run() -> None:
        client = EmbedderClient(base_url="http://stub", backend="tei")
        client._http = httpx.AsyncClient(transport=_mock_transport(handler))
        try:
            vecs = await client.embed(["one", "two"])
            assert len(vecs) == 2
            assert all(len(v) == 3 for v in vecs)
        finally:
            await client._http.aclose()

    asyncio.run(_run())


def test_embedder_handles_empty() -> None:
    async def _run() -> None:
        client = EmbedderClient(base_url="http://stub")
        assert await client.embed([]) == []

    asyncio.run(_run())


def test_reranker_aligns_scores_to_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        # Return out-of-order scores to verify the client re-aligns.
        return httpx.Response(
            200,
            json=[
                {"index": 2, "score": 0.9},
                {"index": 0, "score": 0.4},
                {"index": 1, "score": 0.1},
            ],
        )

    async def _run() -> None:
        client = RerankerClient(base_url="http://stub")
        client._http = httpx.AsyncClient(transport=_mock_transport(handler))
        try:
            scores = await client.rerank("q", ["a", "b", "c"])
            assert scores == [0.4, 0.1, 0.9]
        finally:
            await client._http.aclose()

    asyncio.run(_run())


def test_reranker_empty_candidates() -> None:
    async def _run() -> None:
        client = RerankerClient(base_url="http://stub")
        assert await client.rerank("q", []) == []

    asyncio.run(_run())
