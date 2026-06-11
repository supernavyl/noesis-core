"""Ingest a corpus of your own into NOESIS.

This is the canonical template for adding a custom knowledge source. Point it at
any directory of text-like files (markdown, source code, PDFs converted to text,
plain text, …) and NOESIS will chunk, embed, dedup, and index them through the
same hybrid-retrieval pipeline the built-in sources use.

A source is just an object that yields `IngestionItem`s. `WorkspaceSource` is the
batteries-included one for "a directory on disk"; to ingest from an API, a
database, or a custom format, implement the `IngestionSource` protocol and hand
your instance to `pipeline.run_source(...)` exactly the same way.

Usage:
    python scripts/ingest_example.py /path/to/your/docs --tag my-corpus
    python scripts/ingest_example.py /path/to/docs /path/to/more --tag my-corpus --dry-run
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from noesis.knowledge.ingestion.pipeline import IngestionPipeline
from noesis.knowledge.ingestion.workspace import WorkspaceSource

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def main(
    roots: list[str] = typer.Argument(..., help="One or more directories to ingest."),
    tag: str = typer.Option("custom", "--tag", help="Label/source tag for these chunks."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count files without indexing."),
) -> None:
    """Ingest one or more directories as a single tagged corpus."""
    asyncio.run(_run(roots=roots, tag=tag, dry_run=dry_run))


async def _run(roots: list[str], tag: str, dry_run: bool) -> None:
    # A WorkspaceSource turns a set of directories into a stream of IngestionItems.
    source = WorkspaceSource(roots, label=tag)

    if dry_run:
        count = 0
        async for _ in source.fetch_new():
            count += 1
        console.print(f"[bold]Dry run[/bold]: {count} files would be ingested as '{tag}'.")
        return

    # The pipeline owns chunking, embedding (dense + SPLADE sparse), dedup, and the
    # write into the vector store. `.default()` wires the standard model slots.
    pipeline = IngestionPipeline.default()
    console.print(f"[bold cyan]Ingesting[/bold cyan] '{tag}' from: {', '.join(roots)}")

    stats = await pipeline.run_source(source)

    console.print(
        f"  fetched={stats.fetched} "
        f"written={stats.chunks_written} "
        f"dup={stats.skipped_duplicate} "
        f"err={stats.errors}"
    )
    console.print("[green]Done.[/green] Query it via the API or the retrieval client.")


if __name__ == "__main__":
    app()
