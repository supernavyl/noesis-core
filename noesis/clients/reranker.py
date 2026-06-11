"""Reranker client — talks to TEI rerank endpoint (BGE-Reranker-v2-Gemma by default).

TEI exposes:
  POST /rerank body: {"query": str, "texts": [strs], "truncate": true}
  → returns [{"index": int, "score": float}, ...] sorted by score desc.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from noesis.core.config import settings


class RerankerClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 60.0) -> None:
        self.base_url = (base_url or settings().rerank_service_url).rstrip("/")
        self.timeout_s = timeout_s
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> RerankerClient:
        self._http = httpx.AsyncClient(timeout=self.timeout_s)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
        return self._http

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Return scores aligned to the input candidate order."""
        if not candidates:
            return []
        http = await self._ensure()
        resp = await http.post(
            f"{self.base_url}/rerank",
            json={"query": query, "texts": list(candidates), "truncate": True},
        )
        resp.raise_for_status()
        items = resp.json()
        scores = [0.0] * len(candidates)
        for item in items:
            idx = item["index"]
            if 0 <= idx < len(scores):
                scores[idx] = float(item["score"])
        return scores

    async def health(self) -> bool:
        http = await self._ensure()
        try:
            r = await http.get(f"{self.base_url}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False
