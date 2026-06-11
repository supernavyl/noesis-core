"""Activation steering — surgical behavioral edits without retraining.

Given a target direction in the residual stream (e.g. "more cautious", "less refusal"),
project activations onto/away from that direction at inference time. Wires through TransformerLens
hooks in P6.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SteeringDirection:
    name: str
    layer: int
    vector_path: str
    coefficient: float = 1.0


class Steerer:
    def __init__(self) -> None:
        self.active: list[SteeringDirection] = []

    def add(self, direction: SteeringDirection) -> None:
        self.active.append(direction)

    def clear(self) -> None:
        self.active.clear()

    def apply(self) -> None:
        """Register forward hooks on the model. Implemented in P6."""
        raise NotImplementedError("Steering hook installation is implemented in P6.")
