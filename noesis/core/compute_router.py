"""Compute router — assigns models to GPUs and manages hot-swap on the 4090.

Strategy:
- 4090 (24GB): one active large model at a time. Hot-swap reasoner↔coder when needed.
- 3060 (12GB): always-on embedder + reranker + small critic.
- Long-context synthesizer: llama.cpp CPU+GPU offload, time-multiplexed during dream cycle only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from noesis.core.model_registry import ModelSpec


@dataclass
class GPUSlot:
    device: int
    vram_gb: float
    loaded: ModelSpec | None = None
    in_use: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ComputeRouter:
    def __init__(self) -> None:
        self.gpus = {
            0: GPUSlot(device=0, vram_gb=24),  # 4090
            1: GPUSlot(device=1, vram_gb=12),  # 3060
        }

    async def acquire(self, spec: ModelSpec) -> int:
        """Acquire a GPU slot for the given model. Hot-swaps if needed.

        Returns the device index where the model is loaded.
        """
        target_device = self._target_device_for(spec)
        slot = self.gpus[target_device]

        async with slot.lock:
            if slot.loaded and slot.loaded.id == spec.id:
                slot.in_use = True
                return target_device

            # Need to swap
            if slot.loaded:
                await self._unload(target_device)
            await self._load(target_device, spec)
            slot.in_use = True
            return target_device

    def release(self, device: int) -> None:
        self.gpus[device].in_use = False

    def _target_device_for(self, spec: ModelSpec) -> int:
        # Small (<=12GB) models go to 3060; large to 4090.
        if spec.vram_gb and spec.vram_gb <= 10:
            return 1
        return 0

    async def _load(self, device: int, spec: ModelSpec) -> None:
        """Load a model onto a GPU. Stub — fills in during P2 (vLLM management)."""
        raise NotImplementedError("Model loading wires into vLLM in P2.")

    async def _unload(self, device: int) -> None:
        """Unload current model from GPU. Stub — fills in during P2."""
        raise NotImplementedError("Model unloading wires into vLLM in P2.")
