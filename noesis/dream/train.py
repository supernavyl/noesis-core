"""Unsloth LoRA fine-tuning driver.

Stub-mode here — real Unsloth invocation requires GPU. We emit a JSONL training file and
a runnable training command; the dream cycle launcher invokes it as a subprocess.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from noesis.core.config import dream_config, settings
from noesis.dream.synthesis import QAPair


@dataclass
class TrainArtifacts:
    dataset_path: Path
    adapter_path: Path
    log_path: Path


class LoRATrainer:
    def __init__(self) -> None:
        self.cfg = dream_config()["phases"]["train"]
        self.checkpoint_root = settings().noesis_checkpoint_dir

    def materialize_dataset(self, pairs: list[QAPair]) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        ds_dir = self.checkpoint_root / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        path = ds_dir / f"dream_{ts}.jsonl"
        with path.open("w") as fh:
            for p in pairs:
                fh.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": p.question},
                                {
                                    "role": "assistant",
                                    "content": f"<thinking>\n{p.rationale}\n</thinking>\n\n{p.answer}",
                                },
                            ],
                            "score": p.score,
                            "provenance": p.provenance,
                        }
                    )
                    + "\n"
                )
        return path

    def train(self, dataset: Path) -> TrainArtifacts:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        adapter_path = self.checkpoint_root / "dream_adapters" / ts
        adapter_path.mkdir(parents=True, exist_ok=True)
        log_path = self.checkpoint_root / "logs" / f"train_{ts}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Defer the actual command to a Python helper so we don't shell-out in production code.
        cmd = [
            "python",
            "-m",
            "noesis.dream.train_runner",
            "--dataset",
            str(dataset),
            "--out",
            str(adapter_path),
            "--rank",
            str(self.cfg["rank"]),
            "--alpha",
            str(self.cfg["alpha"]),
            "--lr",
            str(self.cfg["learning_rate"]),
            "--epochs",
            str(self.cfg["epochs"]),
            "--batch-size",
            str(self.cfg["batch_size"]),
        ]
        with log_path.open("w") as logfh:
            subprocess.run(cmd, stdout=logfh, stderr=subprocess.STDOUT, check=False)
        return TrainArtifacts(dataset_path=dataset, adapter_path=adapter_path, log_path=log_path)
