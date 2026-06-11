"""ArXiv ingestor — daily batch via the arxiv Python API.

Pulls papers from configured categories, filters by date, fetches PDFs, extracts full text,
and yields :class:`IngestionItem` per paper.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import arxiv
import httpx
from loguru import logger

from noesis.core.config import ingestion_config
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource
from noesis.knowledge.ingestion.pdf_fetcher import PDFFetcher


class ArxivSource(IngestionSource):
    name = "arxiv"

    def __init__(self, fetch_full_text: bool = True) -> None:
        cfg = ingestion_config()["sources"]["arxiv"]
        self.categories = cfg["categories"]
        self.bootstrap_days = cfg.get("bootstrap_days", 7)
        self.priority_authors = set(cfg.get("priority_authors", []))
        self.fetch_full_text = fetch_full_text
        self.pdf_fetcher = PDFFetcher() if fetch_full_text else None

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        cutoff = datetime.utcnow() - timedelta(days=self.bootstrap_days)
        query = " OR ".join(f"cat:{c}" for c in self.categories)

        search = arxiv.Search(
            query=query,
            max_results=500,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
        loop = asyncio.get_event_loop()

        def _materialize() -> list[arxiv.Result]:
            return list(client.results(search))

        results = await loop.run_in_executor(None, _materialize)

        async with httpx.AsyncClient(timeout=60) as http:
            for paper in results:
                if paper.published.replace(tzinfo=None) < cutoff:
                    break

                authors = [a.name for a in paper.authors]
                tags = ["arxiv"] + list(paper.categories)
                if self.priority_authors & set(authors):
                    tags.append("priority_author")

                text = paper.summary
                if self.fetch_full_text and self.pdf_fetcher and paper.pdf_url:
                    try:
                        full = await self.pdf_fetcher.fetch_and_extract(paper.pdf_url, http=http)
                        if full and len(full) > len(text):
                            # Keep abstract as preamble for context
                            text = f"# {paper.title}\n\n## Abstract\n{paper.summary}\n\n## Full Text\n{full}"
                            tags.append("full_text")
                    except Exception as exc:
                        logger.warning("arxiv pdf fetch failed for {}: {}", paper.entry_id, exc)

                yield IngestionItem(
                    source="arxiv",
                    source_id=paper.entry_id,
                    url=paper.pdf_url,
                    title=paper.title.strip(),
                    text=text,
                    published_at=paper.published.isoformat(),
                    authors=authors,
                    tags=tags,
                )
