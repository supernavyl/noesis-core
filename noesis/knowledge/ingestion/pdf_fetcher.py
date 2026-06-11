"""PDF fetcher — downloads + extracts text from URLs (used by arxiv full-text ingestion)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from noesis.core.config import settings


class PDFFetcher:
    """Caches downloaded PDFs to disk to avoid re-downloading on re-runs."""

    def __init__(self, cache_dir: Path | None = None, max_bytes: int = 25 * 1024 * 1024) -> None:
        self.cache_dir = cache_dir or (settings().noesis_corpus_dir / "arxiv" / "pdfs")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.pdf"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def fetch(self, url: str, http: httpx.AsyncClient) -> bytes | None:
        cached = self._cache_path(url)
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes()

        resp = await http.get(url, follow_redirects=True, timeout=60)
        if resp.status_code != 200:
            return None
        if int(resp.headers.get("content-length", 0)) > self.max_bytes:
            logger.warning(
                "skip oversized PDF: {} bytes from {}", resp.headers.get("content-length"), url
            )
            return None
        data = resp.content
        if len(data) > self.max_bytes:
            return None
        cached.write_bytes(data)
        return data

    async def fetch_and_extract(
        self, url: str, http: httpx.AsyncClient | None = None
    ) -> str | None:
        own = http is None
        if own:
            http = httpx.AsyncClient(timeout=60)
        try:
            data = await self.fetch(url, http)  # type: ignore[arg-type]
            if not data:
                return None
            return await asyncio.to_thread(_extract_pdf_text, data)
        finally:
            if own:
                await http.aclose()  # type: ignore[union-attr]


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF byte buffer using PyMuPDF."""
    import pymupdf

    parts: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts).strip()
