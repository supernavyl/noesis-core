"""Eval gate — holdout regression check before merging a dream-cycle adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class EvalMetrics:
    accuracy: float = 0.0
    faithfulness: float = 0.0
    calibration: float = 0.0
    custom: dict[str, float] = field(default_factory=dict)

    def passes(self, baseline: EvalMetrics, regression_pct: float = 1.0) -> bool:
        for name in ("accuracy", "faithfulness", "calibration"):
            base = getattr(baseline, name)
            mine = getattr(self, name)
            if base > 0 and (base - mine) / base * 100 > regression_pct:
                return False
        return True


@dataclass
class EvalReport:
    baseline: EvalMetrics
    candidate: EvalMetrics
    passed: bool
    failures: list[str] = field(default_factory=list)


class HoldoutEvaluator:
    def __init__(self, holdout_path: Path, model: LLMClient) -> None:
        self.holdout = self._load(holdout_path)
        self.model = model

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]

    async def evaluate(self) -> EvalMetrics:
        if not self.holdout:
            return EvalMetrics()

        correct = 0
        for ex in self.holdout:
            response = await self.model.complete(ex["question"])
            if _exact_or_contains(response, ex["answer"]):
                correct += 1
        acc = correct / len(self.holdout)
        return EvalMetrics(accuracy=acc, faithfulness=acc, calibration=acc)


def _exact_or_contains(response: str, gold: str) -> bool:
    return gold.strip().lower() in response.lower()
