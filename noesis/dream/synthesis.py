"""STaR-style Q&A synthesis.

Given the semantic store, generate reasoning-rich Q&A pairs that the model can then be fine-tuned on.
Zelikman et al. 2022, extended with chain-of-thought traces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class QAPair:
    question: str
    rationale: str
    answer: str
    provenance: list[str]
    score: float | None = None


SYNTHESIS_PROMPT = """From the following knowledge excerpt, generate one high-quality reasoning Q&A pair.

EXCERPT (source: {source}):
{excerpt}

Output exactly:
QUESTION: <a hard, specific question whose answer requires synthesis, not memorization>
RATIONALE: <step-by-step chain-of-thought>
ANSWER: <the final answer>

Constraints:
- The question must not be answerable by quoting a single sentence.
- The rationale must show reasoning, not just restate the answer.
- If the excerpt is too thin to support a reasoning question, output only: SKIP
"""


class Synthesizer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(self, excerpts: list[dict[str, Any]], n: int) -> list[QAPair]:
        pairs: list[QAPair] = []
        sample = random.sample(excerpts, min(n, len(excerpts)))
        for ex in sample:
            raw = await self.llm.complete(
                SYNTHESIS_PROMPT.format(
                    source=ex.get("source", "?"),
                    excerpt=ex.get("text", "")[:3000],
                )
            )
            if "SKIP" in raw and "QUESTION:" not in raw:
                continue
            pair = _parse_pair(raw)
            if pair:
                pair.provenance = [str(ex.get("source_id", ""))]
                pairs.append(pair)
        return pairs


def _parse_pair(raw: str) -> QAPair | None:
    if "QUESTION:" not in raw or "ANSWER:" not in raw:
        return None
    try:
        q = raw.split("QUESTION:", 1)[1].split("RATIONALE:", 1)[0].strip()
        r = raw.split("RATIONALE:", 1)[1].split("ANSWER:", 1)[0].strip()
        a = raw.split("ANSWER:", 1)[1].strip()
        if not q or not a:
            return None
        return QAPair(question=q, rationale=r, answer=a, provenance=[])
    except (IndexError, ValueError):
        return None
