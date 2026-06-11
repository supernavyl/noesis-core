"""Ingest entire projects into NOESIS — all source code + docs.

Edit the PROJECTS list below to point at the repositories you want indexed, then:

    python scripts/ingest_workspace.py
    python scripts/ingest_workspace.py --dry-run

For a single ad-hoc corpus, see scripts/ingest_example.py instead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from noesis.knowledge.ingestion.pipeline import IngestionPipeline
from noesis.knowledge.ingestion.workspace import WorkspaceSource

HOME = Path.home()
app = typer.Typer(no_args_is_help=False)
console = Console()

# Edit this list to point at the repositories you want indexed.
# Each entry is (absolute_path, label). Labels become the source tag on each chunk.
PROJECTS: list[tuple[str, str]] = [
    # AI corpus — inference engines
    (str(HOME / "repos/ai-corpus/vllm"), "vllm"),
    (str(HOME / "repos/ai-corpus/sglang"), "sglang"),
    (str(HOME / "repos/ai-corpus/flashinfer"), "flashinfer"),
    # AI corpus — Rust ML
    (str(HOME / "repos/ai-corpus/mistral.rs"), "mistral-rs"),
    (str(HOME / "repos/ai-corpus/candle"), "candle"),
    (str(HOME / "repos/ai-corpus/burn"), "burn"),
    # AI corpus — coding agents
    (str(HOME / "repos/ai-corpus/OpenHands"), "openhands"),
    (str(HOME / "repos/ai-corpus/aider"), "aider"),
    # AI corpus — agent frameworks
    (str(HOME / "repos/ai-corpus/langgraph"), "langgraph"),
    (str(HOME / "repos/ai-corpus/pydantic-ai"), "pydantic-ai"),
    # AI corpus — RAG + memory
    (str(HOME / "repos/ai-corpus/graphrag"), "graphrag"),
    (str(HOME / "repos/ai-corpus/cognee"), "cognee"),
]


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    asyncio.run(_run(dry_run=dry_run))


async def _run(dry_run: bool) -> None:
    if dry_run:
        console.print("\n[bold]Dry run[/bold]\n")
        total = 0
        for path, label in PROJECTS:
            src = WorkspaceSource([path], label=label)
            count = 0
            async for _ in src.fetch_new():
                count += 1
            console.print(f"  [cyan]{label:20}[/cyan] {count:4d} files   {path}")
            total += count
        console.print(f"\n[bold]Total: {total} files[/bold]")
        return

    pipeline = IngestionPipeline.default()
    all_stats = []
    for path, label in PROJECTS:
        console.print(f"\n[bold cyan]► {label}[/bold cyan]  {path}")
        src = WorkspaceSource([path], label=label)
        stats = await pipeline.run_source(src)
        all_stats.append((label, stats))
        console.print(
            f"  fetched={stats.fetched} written={stats.chunks_written} dup={stats.skipped_duplicate} err={stats.errors}"
        )

    table = Table(title="Complete")
    table.add_column("Project")
    table.add_column("Fetched", justify="right")
    table.add_column("Written", justify="right")
    table.add_column("Dup", justify="right")
    for label, s in all_stats:
        table.add_row(label, str(s.fetched), str(s.chunks_written), str(s.skipped_duplicate))
    console.print(table)


if __name__ == "__main__":
    app()
