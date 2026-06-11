"""Retrieval benchmark — measures HyDE + Self-RAG impact on a small golden set.

Runs four configurations and reports recall@k and skip-rate side-by-side:
  A. baseline   — no HyDE, no Self-RAG gate
  B. hyde       — HyDE expansion only
  C. self_rag   — Self-RAG gate only (skips trivial queries)
  D. full       — HyDE + Self-RAG gate (production default)

Usage:
    python scripts/benchmark_retrieval.py [--golden PATH] [--k 8]

Golden-set format (JSONL):
    {"query": "...", "relevant_chunk_ids": ["id1", "id2"], "is_trivial": false}
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(no_args_is_help=False)

_GOLDEN_DEFAULT = Path(__file__).parent.parent / "data" / "golden_retrieval.jsonl"

_BUILTIN_GOLDEN: list[dict[str, Any]] = [
    {
        "query": "hi",
        "relevant_chunk_ids": [],
        "is_trivial": True,
    },
    {
        "query": "What is Flash Attention?",
        "relevant_chunk_ids": [],
        "is_trivial": False,
    },
    {
        "query": "How does SPLADE sparse retrieval work?",
        "relevant_chunk_ids": [],
        "is_trivial": False,
    },
    {
        "query": "gradient descent",
        "relevant_chunk_ids": [],
        "is_trivial": False,
    },
    {
        "query": "hello",
        "relevant_chunk_ids": [],
        "is_trivial": True,
    },
]


@dataclass
class BenchResult:
    config: str
    queries: int = 0
    skipped: int = 0
    retrieved: int = 0
    recall_sum: float = 0.0
    latency_ms_sum: float = 0.0
    errors: int = 0
    citation_counts: list[int] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        return self.skipped / self.queries if self.queries else 0.0

    @property
    def recall_at_k(self) -> float:
        if self.retrieved == 0:
            return 0.0
        return self.recall_sum / self.retrieved

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_sum / self.queries if self.queries else 0.0

    @property
    def avg_citations(self) -> float:
        if not self.citation_counts:
            return 0.0
        return sum(self.citation_counts) / len(self.citation_counts)


async def _run_config(
    examples: list[dict[str, Any]],
    k: int,
    use_hyde: bool,
    use_gate: bool,
) -> BenchResult:
    from noesis.knowledge.retrieval import HybridRetriever
    from noesis.reasoning.citations import CitationTracker

    config_name = ("hyde+" if use_hyde else "") + ("gate" if use_gate else "baseline")
    if use_hyde and not use_gate:
        config_name = "hyde"
    elif not use_hyde and use_gate:
        config_name = "self_rag"
    elif use_hyde and use_gate:
        config_name = "full"

    result = BenchResult(config=config_name)

    try:
        retriever = HybridRetriever.default()
    except Exception as exc:
        console.print(f"[yellow]Cannot connect to retrieval stack: {exc}[/yellow]")
        console.print("[yellow]Running in dry-run mode (no actual retrieval).[/yellow]")
        retriever = None

    gate = None
    if use_gate:
        try:
            from noesis.clients.llm import VLLMClient
            from noesis.reasoning.self_rag import RetrievalGate

            gate = RetrievalGate(judge=VLLMClient(served_model="reasoner.small"))
        except Exception:
            gate = None

    for ex in examples:
        query: str = ex["query"]
        relevant: set[str] = set(ex.get("relevant_chunk_ids", []))
        result.queries += 1

        t0 = time.monotonic()

        # Self-RAG gate check
        if gate is not None:
            try:
                decision = await gate.decide(query)
                if not decision.retrieve:
                    result.skipped += 1
                    result.latency_ms_sum += (time.monotonic() - t0) * 1000
                    result.citation_counts.append(0)
                    continue
            except Exception:
                pass

        # Actual retrieval
        if retriever is None:
            # dry-run: simulate a hit
            result.retrieved += 1
            result.latency_ms_sum += (time.monotonic() - t0) * 1000
            result.citation_counts.append(0)
            continue

        tracker = CitationTracker()
        try:
            hits = await retriever.search(query=query, k_final=k, use_hyde=use_hyde)
            result.retrieved += 1
            elapsed = (time.monotonic() - t0) * 1000
            result.latency_ms_sum += elapsed

            hit_ids = {h.payload.get("chunk_id", "") for h in hits}
            if relevant:
                recall = len(relevant & hit_ids) / len(relevant)
                result.recall_sum += recall

            # Register citations
            for i, h in enumerate(hits, 1):
                chunk_id = h.payload.get("chunk_id", str(i))
                tracker.register(
                    chunk_id,
                    h.payload.get("title", ""),
                    h.payload.get("url", ""),
                    h.payload.get("source", "?"),
                )
            result.citation_counts.append(len(tracker.citations()))

        except Exception as exc:
            result.errors += 1
            result.latency_ms_sum += (time.monotonic() - t0) * 1000
            result.citation_counts.append(0)

    return result


@app.command()
def main(
    golden: Path = typer.Option(None, "--golden", help="JSONL golden set"),
    k: int = typer.Option(8, "--k", help="k_final for retrieval"),
    configs: str = typer.Option(
        "A,B,C,D", "--configs", help="Comma-separated: A=baseline B=hyde C=self_rag D=full"
    ),
) -> None:
    """Benchmark HyDE + Self-RAG against a golden retrieval set."""

    if golden and golden.exists():
        examples = [json.loads(l) for l in golden.read_text().splitlines() if l.strip()]
        console.print(f"Loaded {len(examples)} examples from {golden}")
    else:
        examples = _BUILTIN_GOLDEN
        console.print(
            f"[yellow]Using built-in {len(examples)}-example golden set (no file at {golden or _GOLDEN_DEFAULT}).[/yellow]"
        )

    run_map = {
        "A": (False, False),
        "B": (True, False),
        "C": (False, True),
        "D": (True, True),
    }
    selected = [c.strip().upper() for c in configs.split(",")]

    async def _all() -> list[BenchResult]:
        tasks = []
        for key in selected:
            if key not in run_map:
                console.print(f"[red]Unknown config {key!r}, skipping[/red]")
                continue
            use_hyde, use_gate = run_map[key]
            tasks.append(_run_config(examples, k, use_hyde, use_gate))
        return await asyncio.gather(*tasks)

    results = asyncio.run(_all())

    t = Table(
        "config",
        "queries",
        "skipped",
        "skip%",
        "retrieved",
        "recall@k",
        "avg_cites",
        "avg_ms",
        "errors",
    )
    for r in results:
        t.add_row(
            r.config,
            str(r.queries),
            str(r.skipped),
            f"{r.skip_rate:.0%}",
            str(r.retrieved),
            f"{r.recall_at_k:.3f}" if r.retrieved else "n/a",
            f"{r.avg_citations:.1f}",
            f"{r.avg_latency_ms:.0f}",
            str(r.errors),
        )
    console.print(t)


if __name__ == "__main__":
    app()
