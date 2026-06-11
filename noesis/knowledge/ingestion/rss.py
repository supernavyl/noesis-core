"""RSS / Atom feed ingestor — high-signal blogs (Lilian Weng, Karpathy, Anthropic, …)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import feedparser
import httpx
import trafilatura

from noesis.core.config import ingestion_config
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource


class RSSSource(IngestionSource):
    name = "rss"

    def __init__(self) -> None:
        cfg = ingestion_config()["sources"]["rss"]
        self.feeds: list[str] = cfg["feeds"]

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as http:
            for feed_url in self.feeds:
                try:
                    feed_resp = await http.get(feed_url)
                    feed = feedparser.parse(feed_resp.text)
                except Exception:
                    continue

                for entry in feed.entries[:50]:
                    url = entry.get("link")
                    if not url:
                        continue
                    try:
                        page_resp = await http.get(url)
                        text = await asyncio.to_thread(
                            trafilatura.extract, page_resp.text, include_links=False
                        )
                    except Exception:
                        text = entry.get("summary", "")
                    if not text:
                        continue
                    yield IngestionItem(
                        source="rss",
                        source_id=url,
                        url=url,
                        title=entry.get("title", "untitled"),
                        text=text,
                        published_at=entry.get("published"),
                        authors=[entry.get("author")] if entry.get("author") else [],
                        tags=["rss", _feed_tag(feed_url)],
                    )


def _feed_tag(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]
