"""Refusal direction drift monitor.

Track the magnitude/direction of the abliterated model's refusal direction over time. If it
shifts more than ``refusal_drift_threshold`` after a dream-cycle merge, halt and alert.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefusalSnapshot:
    captured_at: str
    layer: int
    direction_norm: float
    sample_size: int


class RefusalMonitor:
    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold
        self.baseline: RefusalSnapshot | None = None

    def set_baseline(self, snapshot: RefusalSnapshot) -> None:
        self.baseline = snapshot

    def compare(self, current: RefusalSnapshot) -> tuple[bool, float]:
        if not self.baseline:
            return True, 0.0
        delta = abs(current.direction_norm - self.baseline.direction_norm) / max(
            self.baseline.direction_norm, 1e-9
        )
        return delta <= self.threshold, delta
