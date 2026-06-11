"""Sparse autoencoders on residual stream activations.

Train SAEs per layer with SAELens. Each feature is then labeled with a judge model so we can
ask things like "does feature 12345 correlate with refusal?"

Heavy module — fully implemented in P6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SAEConfig:
    base_model: str
    layer: int
    expansion_factor: int = 16
    l1_coefficient: float = 1e-3
    learning_rate: float = 3e-4
    train_steps: int = 50_000


class SAERunner:
    def __init__(self, config: SAEConfig, out_dir: Path) -> None:
        self.config = config
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Path:
        """Train an SAE on the configured layer. Wires SAELens in P6."""
        raise NotImplementedError("SAE training is implemented in P6.")
