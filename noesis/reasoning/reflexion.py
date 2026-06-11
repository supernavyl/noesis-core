"""Reflexion — verbal reinforcement after failed attempts.

Shinn et al. 2023. After a failed run, the model generates a verbal self-critique that is
prepended to the next attempt as additional context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class ReflectionMemory:
    attempts: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)


REFLECT_PROMPT = """Previous attempt failed:

ATTEMPT:
{attempt}

OUTCOME: {outcome}

Reflect on what went wrong. Write a short, concrete self-critique (2-4 sentences) that — if
prepended to the next attempt — would prevent the same mistake. Do not restate the question.
"""


class Reflector:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def reflect(self, attempt: str, outcome: str) -> str:
        return await self.llm.complete(REFLECT_PROMPT.format(attempt=attempt, outcome=outcome))

    @staticmethod
    def prepend_to_prompt(reflections: list[str], base_prompt: str) -> str:
        if not reflections:
            return base_prompt
        rec = "\n".join(f"- {r}" for r in reflections)
        return f"Lessons from prior attempts:\n{rec}\n\n{base_prompt}"
