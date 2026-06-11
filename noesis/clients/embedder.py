"""Embedder client — supports Ollama (default) and TEI backends.

Backend selection comes from `settings().backend`. Both paths share the same surface so
callers don't care which is running.

Ollama:  POST /api/embed   {"model": "qwen3-embedding:8b", "input": [strs]}  → {"embeddings": [[…]]}
TEI:     POST /embed       {"inputs": [strs], "truncate": true, "normalize": true}  → [[…]]
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from noesis.core.config import models_config, settings


class EmbedderClient:
    """Async client. Switches request shape based on `settings().backend`."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        backend: str | None = None,
        model: str | None = None,
    ) -> None:
        s = settings()
        self.backend = backend or s.backend
        self.base_url = (base_url or s.embed_service_url).rstrip("/")
        self.timeout_s = timeout_s
        self.model = model or self._resolve_model()
        self._http: httpx.AsyncClient | None = None

    def _resolve_model(self) -> str:
        try:
            cfg = models_config()["embedder"]["primary"]
            if self.backend == "tei":
                return cfg.get("alt_ids", {}).get("tei", cfg["id"])
            return cfg["id"]
        except Exception:
            return "qwen3-embedding:8b"

    async def __aenter__(self) -> EmbedderClient:
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
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        http = await self._ensure()
        results: list[list[float]] = []
        for batch in _batched(list(texts), 32):
            if self.backend == "ollama":
                resp = await http.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch, "truncate": True},
                )
                resp.raise_for_status()
                data = resp.json()
                vecs = data.get("embeddings")
                if not isinstance(vecs, list):
                    raise RuntimeError(f"ollama embed bad shape: {data}")
                results.extend(vecs)
            else:  # tei
                resp = await http.post(
                    f"{self.base_url}/embed",
                    json={"inputs": batch, "truncate": True, "normalize": True},
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    results.extend(data)
                else:
                    raise RuntimeError(f"tei embed error: {data}")
        return results

    async def health(self) -> bool:
        http = await self._ensure()
        try:
            if self.backend == "ollama":
                r = await http.get(f"{self.base_url}/api/tags")
            else:
                r = await http.get(f"{self.base_url}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False


def _batched(xs: list[str], size: int) -> list[list[str]]:
    return [xs[i : i + size] for i in range(0, len(xs), size)]


async def quick_embed(texts: list[str]) -> list[list[float]]:
    """One-shot helper for scripts."""
    async with EmbedderClient() as e:
        return await e.embed(texts)


if __name__ == "__main__":
    import sys

    out = asyncio.run(quick_embed(sys.argv[1:] or ["hello"]))
    print(f"embedded {len(out)} texts, dim={len(out[0]) if out else 0}")
