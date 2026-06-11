"""HuggingFace ingestor — trending models + daily papers."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource


class HuggingFaceSource(IngestionSource):
    name = "huggingface"

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        async with httpx.AsyncClient(timeout=20) as client:
            # Trending models
            r = await client.get("https://huggingface.co/api/models?sort=trending&limit=30")
            for m in r.json():
                model_id = m.get("modelId") or m.get("id")
                if not model_id:
                    continue
                try:
                    card = await client.get(f"https://huggingface.co/{model_id}/raw/main/README.md")
                    text = card.text if card.status_code == 200 else m.get("description", "")
                except Exception:
                    text = m.get("description", "")
                yield IngestionItem(
                    source="huggingface",
                    source_id=f"model/{model_id}",
                    url=f"https://huggingface.co/{model_id}",
                    title=model_id,
                    text=text,
                    tags=["huggingface", "model", *m.get("tags", [])[:5]],
                )

            # Daily papers
            r = await client.get("https://huggingface.co/api/daily_papers")
            for p in r.json():
                arxiv_id = p.get("paper", {}).get("id")
                if not arxiv_id:
                    continue
                paper = p.get("paper", {})
                yield IngestionItem(
                    source="huggingface",
                    source_id=f"paper/{arxiv_id}",
                    url=f"https://huggingface.co/papers/{arxiv_id}",
                    title=paper.get("title", "untitled"),
                    text=paper.get("summary", ""),
                    authors=[a.get("name", "") for a in paper.get("authors", [])],
                    tags=["huggingface", "daily_paper"],
                )
