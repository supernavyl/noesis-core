"""Elastic Weight Consolidation — prevent catastrophic forgetting across dream cycles.

Kirkpatrick et al. 2017. Maintain a Fisher information matrix over a reference dataset, then
penalize updates that move important weights too far. Implementation lives in train_runner;
this module holds the helpers + state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FisherInfo:
    """Per-parameter Fisher diagonal estimate.

    Stored as a side-by-side JSON manifest pointing to a safetensors file with the actual values
    (which are too big to JSON-encode).
    """

    manifest_path: Path
    values_path: Path
    reference_examples: int
    base_model: str
    captured_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "values_path": str(self.values_path),
            "reference_examples": self.reference_examples,
            "base_model": self.base_model,
            "captured_at": self.captured_at,
        }


def load_fisher(path: Path) -> FisherInfo | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return FisherInfo(**{k: Path(v) if k.endswith("_path") else v for k, v in data.items()})
