"""Model registry — resolves slot names to concrete model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from noesis.core.config import models_config


@dataclass(frozen=True)
class ModelSpec:
    id: str
    slot: str
    role: str  # primary | fallback | small | large
    type: str = "dense"  # dense | moe
    vram_gb: float | None = None
    context: int = 32768
    notes: str = ""
    raw: dict[str, Any] | None = None


class ModelRegistry:
    """Loads model slot config and serves :class:`ModelSpec` lookups."""

    def __init__(self) -> None:
        self._cfg = models_config()

    def get(self, slot: str, role: str = "primary") -> ModelSpec:
        slot_cfg = self._cfg.get(slot)
        if not slot_cfg or role not in slot_cfg:
            raise KeyError(f"Unknown slot/role: {slot}.{role}")
        raw = slot_cfg[role]
        return ModelSpec(
            id=raw["id"],
            slot=slot,
            role=role,
            type=raw.get("type", "dense"),
            vram_gb=raw.get("vram_fp8_gb") or raw.get("vram_gb"),
            context=raw.get("context", 32768),
            notes=raw.get("notes", ""),
            raw=raw,
        )

    def all_slots(self) -> list[str]:
        return [
            k for k in self._cfg if isinstance(self._cfg[k], dict) and "primary" in self._cfg[k]
        ]

    def routing(self) -> dict[str, str]:
        return self._cfg.get("routing", {})
