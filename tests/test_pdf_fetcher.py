"""PDFFetcher — cache + size limit smoke tests (no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from noesis.knowledge.ingestion.pdf_fetcher import PDFFetcher


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_pdf_fetcher_caches_to_disk(tmp_path: Path) -> None:
    fake_pdf = b"%PDF-1.4\n%fake\n"
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=fake_pdf, headers={"content-length": str(len(fake_pdf))})

    async def _run() -> None:
        fetcher = PDFFetcher(cache_dir=tmp_path)
        http = httpx.AsyncClient(transport=_mock_transport(handler))
        try:
            a = await fetcher.fetch("https://example.com/x.pdf", http)
            b = await fetcher.fetch("https://example.com/x.pdf", http)
            assert a == fake_pdf
            assert b == fake_pdf
            assert calls["n"] == 1, "second call should hit cache"
        finally:
            await http.aclose()

    asyncio.run(_run())


def test_pdf_fetcher_size_limit(tmp_path: Path) -> None:
    big = b"x" * (50 * 1024 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-length": str(len(big))})

    async def _run() -> None:
        fetcher = PDFFetcher(cache_dir=tmp_path, max_bytes=1024)
        http = httpx.AsyncClient(transport=_mock_transport(handler))
        try:
            out = await fetcher.fetch("https://example.com/big.pdf", http)
            assert out is None
        finally:
            await http.aclose()

    asyncio.run(_run())
