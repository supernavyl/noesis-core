"""Tool registry — capability-described callables the model can invoke."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

ToolFn = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    signature: dict[str, Any]  # JSON-schema parameter shape
    fn: ToolFn
    synthesized: bool = False
    verified: bool = False
    use_count: int = 0


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.signature}
            for t in self.tools.values()
        ]

    async def call(self, name: str, params: dict[str, Any]) -> Any:
        tool = self.tools[name]
        tool.use_count += 1
        return await tool.fn(params)
