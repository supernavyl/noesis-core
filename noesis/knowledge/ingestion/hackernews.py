"""Hacker News ingestor — AI/ML-tagged threads above a points threshold."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from noesis.core.config import ingestion_config
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource

HN_API = "https://hacker-news.firebaseio.com/v0"
ALGOLIA = "https://hn.algolia.com/api/v1/search"


class HackerNewsSource(IngestionSource):
    name = "hackernews"

    def __init__(self) -> None:
        cfg = ingestion_config()["sources"]["hackernews"]
        self.min_points: int = cfg.get("min_points", 100)
        self.keywords: list[str] = cfg.get("keyword_filter", [])

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        async with httpx.AsyncClient(timeout=20) as client:
            for kw in self.keywords:
                resp = await client.get(
                    ALGOLIA,
                    params={
                        "query": kw,
                        "tags": "story",
                        "numericFilters": f"points>={self.min_points}",
                        "hitsPerPage": 30,
                    },
                )
                hits = resp.json().get("hits", [])
                for hit in hits:
                    url = (
                        hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
                    )
                    text = hit.get("story_text") or hit.get("title", "")
                    if hit.get("url"):
                        try:
                            page = await client.get(hit["url"], timeout=10)
                            import trafilatura

                            extracted = await asyncio.to_thread(trafilatura.extract, page.text)
                            if extracted:
                                text = extracted
                        except Exception:
                            pass
                    yield IngestionItem(
                        source="hackernews",
                        source_id=hit["objectID"],
                        url=url,
                        title=hit.get("title", "untitled"),
                        text=text,
                        published_at=hit.get("created_at"),
                        authors=[hit.get("author", "anon")],
                        tags=["hackernews", f"points:{hit.get('points', 0)}", f"kw:{kw}"],
                    )
