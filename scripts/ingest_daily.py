"""Run the daily ingestion pipeline across all enabled sources."""

from __future__ import annotations

import asyncio

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from noesis.core.config import ingestion_config
from noesis.knowledge.ingestion.arxiv import ArxivSource
from noesis.knowledge.ingestion.books import BookSource
from noesis.knowledge.ingestion.github import GitHubSource
from noesis.knowledge.ingestion.hackernews import HackerNewsSource
from noesis.knowledge.ingestion.huggingface import HuggingFaceSource
from noesis.knowledge.ingestion.pipeline import IngestionPipeline, PipelineStats
from noesis.knowledge.ingestion.reddit import RedditSource
from noesis.knowledge.ingestion.rss import RSSSource
from noesis.knowledge.ingestion.youtube import YouTubeSource

app = typer.Typer(no_args_is_help=False)
console = Console()

SOURCE_CLASSES = {
    "arxiv": ArxivSource,
    "github": GitHubSource,
    "reddit": RedditSource,
    "hackernews": HackerNewsSource,
    "rss": RSSSource,
    "youtube": YouTubeSource,
    "books": BookSource,
    "huggingface": HuggingFaceSource,
}


@app.command()
def main(
    only: list[str] = typer.Option([], help="Only run these sources (default: all enabled)"),
    bootstrap: bool = typer.Option(False, "--bootstrap", help="Initial backfill mode"),
) -> None:
    cfg = ingestion_config()["sources"]
    enabled = [
        name for name, src in cfg.items() if src.get("enabled") and (not only or name in only)
    ]

    if bootstrap:
        console.print("[yellow]Bootstrap mode — pulling extended history.[/yellow]")

    async def run() -> list[PipelineStats]:
        pipeline = IngestionPipeline()
        results: list[PipelineStats] = []
        for name in enabled:
            if name not in SOURCE_CLASSES:
                logger.warning("unknown source: {}", name)
                continue
            logger.info("=== {} ===", name)
            src = SOURCE_CLASSES[name]()
            stats = await pipeline.run_source(src)
            results.append(stats)
        return results

    all_stats = asyncio.run(run())

    t = Table("source", "fetched", "written", "duplicate", "errors")
    for s in all_stats:
        t.add_row(
            s.source, str(s.fetched), str(s.chunks_written), str(s.skipped_duplicate), str(s.errors)
        )
    console.print(t)


if __name__ == "__main__":
    app()
