"""NOESIS CLI — `noesis` command entry point."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from noesis.core.config import constitution, models_config
from noesis.knowledge.vector_store import VectorStore

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="NOESIS — self-improving AI engineering specialist.",
)
console = Console()


@app.command()
def status() -> None:
    """Show service status + corpus stats."""

    async def _run() -> None:
        vs = VectorStore()
        try:
            count = await vs.count()
        except Exception as exc:
            console.print(f"[red]Qdrant unreachable: {exc}[/red]")
            return
        console.print(f"[green]✓[/green] corpus chunks: {count:,}")

    asyncio.run(_run())


@app.command()
def models() -> None:
    """Show the model registry."""
    cfg = models_config()
    t = Table("slot", "role", "id", "vram", "notes")
    for slot, roles in cfg.items():
        if not isinstance(roles, dict):
            continue
        for role, spec in roles.items():
            if not isinstance(spec, dict):
                continue
            t.add_row(
                slot,
                role,
                spec.get("id", "?"),
                str(spec.get("vram_fp8_gb") or spec.get("vram_gb") or "-"),
                spec.get("notes", "")[:60],
            )
    console.print(t)


@app.command(name="constitution")
def constitution_cmd() -> None:
    """Print the active constitution."""
    c = constitution()
    console.print(f"[bold]NOESIS Constitution v{c.version}[/bold] (modified {c.last_modified})\n")
    for p in c.principles:
        console.print(f"[cyan]{p['id']}[/cyan] (rank {p['rank']})")
        console.print(f"  {p['statement'].strip()}\n")


@app.command()
def ask(question: str) -> None:
    """Send a query to the local NOESIS API."""
    import httpx

    async def _run() -> None:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post("http://localhost:8000/v1/chat", json={"question": question})
            console.print(r.json())

    asyncio.run(_run())


@app.command()
def search(
    query: str,
    k: int = typer.Option(8, help="Number of final results"),
    k_candidates: int = typer.Option(64, help="Hybrid candidates before rerank"),
    source: str | None = typer.Option(None, help="Filter by source (arxiv, github, …)"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
) -> None:
    """Retrieval-only search over the corpus — no reasoner needed."""
    from noesis.knowledge.retrieval import HybridRetriever

    async def _run() -> None:
        retriever = HybridRetriever.default()
        if no_rerank:
            retriever.reranker = None
        filt = {"source": source} if source else None
        hits = await retriever.search(
            query=query, k_candidates=k_candidates, k_final=k, filter_=filt
        )
        if not hits:
            console.print("[yellow]No matches.[/yellow]")
            return
        t = Table("score", "source", "title", "snippet")
        for h in hits:
            t.add_row(
                f"{h.score:.3f}",
                h.payload.get("source", "?"),
                (h.payload.get("title") or "")[:50],
                h.text[:100].replace("\n", " "),
            )
        console.print(t)

    asyncio.run(_run())


@app.command()
def ingest(
    only: list[str] = typer.Option([], "--only", help="Sources to run (default: all enabled)"),  # noqa: B008  (typer requires Option() in defaults)
    bootstrap: bool = typer.Option(False, "--bootstrap"),  # noqa: B008
) -> None:
    """Run the ingestion pipeline. Wraps scripts/ingest_daily.py logic."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/ingest_daily.py"]
    for s in only:
        cmd += ["--only", s]
    if bootstrap:
        cmd += ["--bootstrap"]
    subprocess.run(cmd, check=False)


@app.command()
def health() -> None:
    """Probe all dependent services and report status."""

    async def _run() -> None:
        from noesis.clients.embedder import EmbedderClient
        from noesis.clients.reranker import RerankerClient

        t = Table("service", "status")

        try:
            async with EmbedderClient() as e:
                ok = await e.health()
        except Exception:
            ok = False
        t.add_row("embedder", "[green]✓[/green]" if ok else "[red]✗[/red]")

        try:
            async with RerankerClient() as r:
                ok = await r.health()
        except Exception:
            ok = False
        t.add_row("reranker", "[green]✓[/green]" if ok else "[red]✗[/red]")

        try:
            from noesis.knowledge.vector_store import VectorStore

            vs = VectorStore()
            count = await vs.count()
            t.add_row("qdrant", f"[green]✓[/green] ({count:,} chunks)")
        except Exception as exc:
            t.add_row("qdrant", f"[red]✗ {exc}[/red]")

        # Reasoner via the configured backend
        try:
            from noesis.clients.llm import VLLMClient

            async with VLLMClient() as llm:
                ok = await llm.health()
            t.add_row(
                f"reasoner ({llm.backend})",
                f"[green]✓[/green] {llm.served_model}" if ok else "[red]✗[/red]",
            )
        except Exception as exc:
            t.add_row("reasoner", f"[red]✗ {exc}[/red]")

        console.print(t)

    asyncio.run(_run())


@app.command()
def slots() -> None:
    """Show resolved model IDs per slot for the active backend."""
    from noesis.clients.llm import resolve_slot
    from noesis.core.config import settings

    backend = settings().backend
    cfg = models_config()
    t = Table("slot", "role", "resolved id", "vram")
    for slot, roles in cfg.items():
        if not isinstance(roles, dict):
            continue
        for role, spec in roles.items():
            if not isinstance(spec, dict) or "id" not in spec:
                continue
            resolved = resolve_slot(f"{slot}.{role}", backend=backend)
            vram = spec.get("vram_gb") or spec.get("vram_fp8_gb") or "-"
            t.add_row(slot, role, resolved, str(vram))
    console.print(f"[bold]Active backend:[/bold] {backend}\n")
    console.print(t)


if __name__ == "__main__":
    app()
