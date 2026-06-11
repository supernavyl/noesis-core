"""Book ingestor — PDF and EPUB → text via PyMuPDF + ebooklib."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from noesis.core.config import ingestion_config
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource


class BookSource(IngestionSource):
    name = "books"

    def __init__(self) -> None:
        cfg = ingestion_config()["sources"]["books"]
        self.dirs: list[Path] = [Path(d).expanduser() for d in cfg.get("directories", [])]
        self.formats: list[str] = cfg.get("formats", ["pdf", "epub"])

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        for d in self.dirs:
            if not d.exists():
                continue
            for path in d.rglob("*"):
                ext = path.suffix.lstrip(".").lower()
                if ext not in self.formats:
                    continue
                try:
                    text = await asyncio.to_thread(_extract_text, path)
                except Exception:
                    continue
                if not text:
                    continue
                yield IngestionItem(
                    source="books",
                    source_id=str(path),
                    url=f"file://{path}",
                    title=path.stem,
                    text=text,
                    tags=["book", ext, d.name],
                )


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        import pymupdf

        doc = pymupdf.open(path)
        parts = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(parts)
    if path.suffix.lower() == ".epub":
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub

        book = epub.read_epub(str(path))
        parts: list[str] = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            html = item.get_content().decode("utf-8", errors="ignore")
            parts.append(BeautifulSoup(html, "lxml").get_text(" "))
        return "\n".join(parts)
    return ""
