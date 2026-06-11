"""Dream cycle orchestrator — runs all phases end-to-end."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import psycopg
from loguru import logger

from noesis.core.config import dream_config, settings
from noesis.dream.eval import EvalMetrics, HoldoutEvaluator
from noesis.dream.filter import JudgeFilter
from noesis.dream.synthesis import Synthesizer
from noesis.dream.train import LoRATrainer
from noesis.memory.consolidation import Consolidator
from noesis.memory.semantic import SemanticStore


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class CycleReport:
    started_at: datetime
    ended_at: datetime | None
    synthetic_pairs: int
    pairs_kept: int
    eval_baseline: EvalMetrics | None
    eval_after: EvalMetrics | None
    merged: bool
    rollback_reason: str | None


class DreamCycle:
    def __init__(
        self,
        synthesizer_model: LLMClient,
        judge_model: LLMClient,
        eval_model_baseline: LLMClient,
        eval_model_candidate: LLMClient,
    ) -> None:
        self.cfg = dream_config()
        self.consolidator = Consolidator(synthesizer=synthesizer_model)
        self.synthesizer = Synthesizer(llm=synthesizer_model)
        self.filter = JudgeFilter(
            judge=judge_model,
            min_score=self.cfg["phases"]["filter"]["min_score"],
            keep_top_pct=self.cfg["phases"]["filter"]["keep_top_pct"] / 100,
        )
        self.trainer = LoRATrainer()
        holdout_path = settings().noesis_data_dir / "holdout" / "ai_engineering_bench.jsonl"
        self.eval_baseline = HoldoutEvaluator(holdout_path=holdout_path, model=eval_model_baseline)
        self.eval_candidate = HoldoutEvaluator(
            holdout_path=holdout_path, model=eval_model_candidate
        )
        self.semantic = SemanticStore()

    async def run(self) -> CycleReport:
        started = datetime.utcnow()
        run_id = await self._log_start(started)

        # NREM
        if self.cfg["phases"]["nrem"]["enabled"]:
            cons = await self.consolidator.run(
                window_days=self.cfg["phases"]["nrem"]["episodic_window_days"]
            )
            logger.info("NREM: facts written = {}", cons.facts_written)

        # Synthesis
        n = self.cfg["phases"]["synthesis"]["n_qa_pairs"]
        # Sample broadly from semantic store
        excerpts = await self.semantic.query(
            "the most important concepts in AI engineering", k=n * 3
        )
        pairs = await self.synthesizer.generate(excerpts=excerpts, n=n)
        logger.info("synthesis: generated {} pairs", len(pairs))

        # Filter
        filt = await self.filter.filter(pairs)
        logger.info("filter: kept {}/{} (mean={:.2f})", len(filt.kept), len(pairs), filt.mean_score)

        if not filt.kept:
            return await self._finalize(
                run_id, started, 0, 0, None, None, False, "no pairs survived filter"
            )

        # Eval baseline
        baseline = await self.eval_baseline.evaluate()
        logger.info("baseline eval: acc={:.3f}", baseline.accuracy)

        # Train
        ds = self.trainer.materialize_dataset(filt.kept)
        artifacts = self.trainer.train(ds)
        logger.info("train: adapter={}", artifacts.adapter_path)

        # Eval candidate
        candidate = await self.eval_candidate.evaluate()
        logger.info("candidate eval: acc={:.3f}", candidate.accuracy)

        regression_pct = self.cfg["phases"]["eval"]["rollback_on_regression_pct"]
        merged = candidate.passes(baseline, regression_pct=regression_pct)
        rollback_reason = None if merged else f"regression > {regression_pct}%"

        return await self._finalize(
            run_id,
            started,
            len(pairs),
            len(filt.kept),
            baseline,
            candidate,
            merged,
            rollback_reason,
        )

    async def _log_start(self, started: datetime) -> int:
        async with await psycopg.AsyncConnection.connect(settings().postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO dream_runs (started_at, status) VALUES (%s, %s) RETURNING id",
                    (started, "running"),
                )
                row = await cur.fetchone()
                await conn.commit()
                return int(row[0]) if row else -1

    async def _finalize(
        self,
        run_id: int,
        started: datetime,
        synthetic: int,
        kept: int,
        baseline: EvalMetrics | None,
        candidate: EvalMetrics | None,
        merged: bool,
        rollback_reason: str | None,
    ) -> CycleReport:
        ended = datetime.utcnow()
        async with await psycopg.AsyncConnection.connect(settings().postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """UPDATE dream_runs SET ended_at=%s, status=%s, synthetic_pairs=%s, pairs_kept=%s,
                                                merged=%s, rollback_reason=%s
                       WHERE id=%s""",
                    (
                        ended,
                        "completed" if merged else "rolled_back",
                        synthetic,
                        kept,
                        merged,
                        rollback_reason,
                        run_id,
                    ),
                )
                await conn.commit()
        return CycleReport(
            started_at=started,
            ended_at=ended,
            synthetic_pairs=synthetic,
            pairs_kept=kept,
            eval_baseline=baseline,
            eval_after=candidate,
            merged=merged,
            rollback_reason=rollback_reason,
        )
