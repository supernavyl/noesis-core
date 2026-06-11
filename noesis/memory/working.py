"""Working memory — current-context buffer, no persistence.

Lives in-process. Holds the active reasoning trace, recent retrievals, and tool outputs
for the duration of a single query.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    capacity: int = 64
    items: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        if self.items.maxlen != self.capacity:
            self.items = deque(self.items, maxlen=self.capacity)

    def add(self, kind: str, content: Any) -> None:
        self.items.append({"kind": kind, "content": content})

    def recent(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is None:
            return list(self.items)
        return [i for i in self.items if i["kind"] == kind]

    def clear(self) -> None:
        self.items.clear()
