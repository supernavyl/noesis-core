"""Manually trigger a dream cycle (otherwise scheduled by cron / systemd timer)."""

from __future__ import annotations

import asyncio
import sys

import typer
from loguru import logger
from rich.console import Console

console = Console()
app = typer.Typer(no_args_is_help=False)


@app.command()
def main(manual: bool = typer.Option(False, "--manual")) -> None:
    """Run one dream cycle end-to-end."""

    async def _run() -> int:
        try:
            from noesis.dream.cycle import DreamCycle
        except ImportError as exc:
            console.print(f"[red]Missing dependencies: {exc}[/red]")
            return 2

        # Wiring of concrete LLM clients is filled in P2 when vLLM is up.
        # Until then, this is a structural smoke test.
        console.print("[yellow]Dream cycle wiring requires P2 (vLLM clients). Stubbed.[/yellow]")
        logger.info("dream cycle stub run — manual={}", manual)
        return 0

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    app()
