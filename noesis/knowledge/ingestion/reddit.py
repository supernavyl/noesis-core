"""Reddit ingestor — top-of-week from configured subreddits."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import praw

from noesis.core.config import ingestion_config, resolve_secret
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource


class RedditSource(IngestionSource):
    name = "reddit"

    def __init__(self) -> None:
        cfg = ingestion_config()["sources"]["reddit"]
        self.subreddits: list[str] = cfg["subreddits"]
        self.min_score: int = cfg.get("min_score", 50)
        self.top_window: str = cfg.get("top_window", "week")
        self._client = praw.Reddit(
            client_id=resolve_secret("REDDIT_CLIENT_ID"),
            client_secret=resolve_secret("REDDIT_CLIENT_SECRET"),
            user_agent=resolve_secret("REDDIT_USER_AGENT") or "noesis-ingest/0.0.1",
        )

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        for sub_name in self.subreddits:
            posts = await asyncio.to_thread(
                lambda s=sub_name: list(self._client.subreddit(s).top(self.top_window, limit=50))
            )
            for post in posts:
                if post.score < self.min_score:
                    continue
                body = (post.selftext or "").strip()
                if not body and not post.url:
                    continue
                text = f"{post.title}\n\n{body}"
                yield IngestionItem(
                    source="reddit",
                    source_id=post.id,
                    url=f"https://reddit.com{post.permalink}",
                    title=post.title,
                    text=text,
                    published_at=str(post.created_utc),
                    authors=[str(post.author) if post.author else "deleted"],
                    tags=["reddit", f"r/{sub_name}", f"score:{post.score}"],
                )
