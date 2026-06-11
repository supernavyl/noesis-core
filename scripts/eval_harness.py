"""Eval harness — runs holdout benchmarks against the current model + last adapter.

Used both for dream-cycle gating and for manual regression checks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from noesis.core.config import settings
from noesis.dream.eval import HoldoutEvaluator

app = typer.Typer(no_args_is_help=False)
console = Console()


@app.command()
def main(
    holdout: Path = typer.Option(None, "--holdout"),
    model_endpoint: str = typer.Option(None, "--model-endpoint"),
) -> None:
    holdout = holdout or settings().noesis_data_dir / "holdout" / "ai_engineering_bench.jsonl"
    if not holdout.exists():
        console.print(f"[red]No holdout file at {holdout}[/red]")
        return

    async def _run() -> None:
        console.print(f"[cyan]Eval harness requires P2 vLLM wiring. Holdout path: {holdout}[/cyan]")
        # Smoke: prove we can read the holdout.
        evaluator = HoldoutEvaluator(holdout_path=holdout, model=None)  # type: ignore[arg-type]
        t = Table("metric", "value")
        t.add_row("examples", str(len(evaluator.holdout)))
        console.print(t)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
