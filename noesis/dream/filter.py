"""Judge filter — score synthetic Q&A pairs and keep only top-K.

The quality of self-training data is the entire game. A bad filter = model fine-tunes on garbage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from noesis.dream.synthesis import QAPair


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


JUDGE_PROMPT = """Rate this question-answer pair for use as a training example.

QUESTION: {q}
RATIONALE: {r}
ANSWER: {a}

Score 0.0-1.0 on each axis:
- DIFFICULTY: how non-trivial is the question?
- CORRECTNESS: is the answer factually correct?
- REASONING: does the rationale actually justify the answer?
- CLARITY: is the answer well-formed and unambiguous?
- USEFULNESS: would training on this improve a reasoning model?

Output JSON only:
{{"difficulty": 0.0, "correctness": 0.0, "reasoning": 0.0, "clarity": 0.0, "usefulness": 0.0, "overall": 0.0}}
"""


@dataclass
class FilteredResult:
    kept: list[QAPair]
    rejected: list[QAPair]
    mean_score: float


class JudgeFilter:
    def __init__(self, judge: LLMClient, min_score: float = 0.8, keep_top_pct: float = 0.1) -> None:
        self.judge = judge
        self.min_score = min_score
        self.keep_top_pct = keep_top_pct

    async def filter(self, pairs: list[QAPair]) -> FilteredResult:
        scored: list[QAPair] = []
        for p in pairs:
            raw = await self.judge.complete(
                JUDGE_PROMPT.format(q=p.question, r=p.rationale, a=p.answer)
            )
            p.score = _extract_overall(raw)
            scored.append(p)

        scored.sort(key=lambda x: x.score or 0.0, reverse=True)
        keep_n = max(1, int(len(scored) * self.keep_top_pct))
        kept = [p for p in scored[:keep_n] if (p.score or 0.0) >= self.min_score]
        rejected = [p for p in scored if p not in kept]
        mean = (sum(p.score or 0.0 for p in scored) / len(scored)) if scored else 0.0
        return FilteredResult(kept=kept, rejected=rejected, mean_score=mean)


def _extract_overall(raw: str) -> float:
    import orjson

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = orjson.loads(raw[start:end])
        return float(data.get("overall", 0.0))
    except (ValueError, KeyError, orjson.JSONDecodeError):
        return 0.0
