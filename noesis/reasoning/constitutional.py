"""L5 — Constitutional self-review.

After the answer is drafted, the model self-critiques it against each constitutional principle
and revises if any principle is violated. Final pass before user-visible output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from noesis.core.config import constitution


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class ConstitutionalReview:
    principle_id: str
    passed: bool
    rationale: str
    revision: str | None


@dataclass
class ConstitutionalResult:
    original: str
    final: str
    reviews: list[ConstitutionalReview]
    revised: bool


REVIEW_PROMPT = """Constitutional review.

Principle:
  ID: {pid}
  Rule: {statement}
  Self-check question: {self_check}

Draft answer to evaluate:
{draft}

Output:
PASSES: <yes|no>
RATIONALE: <one sentence>
REVISION: <full revised answer if PASSES=no, else empty>
"""


class ConstitutionalReviewer:
    def __init__(self, reviewer: LLMClient) -> None:
        self.reviewer = reviewer
        self._constitution = constitution()

    async def review(self, draft: str) -> ConstitutionalResult:
        current = draft
        reviews: list[ConstitutionalReview] = []
        revised_any = False

        for principle in self._constitution.principles:
            raw = await self.reviewer.complete(
                REVIEW_PROMPT.format(
                    pid=principle["id"],
                    statement=principle["statement"],
                    self_check=principle.get("self_check", ""),
                    draft=current,
                )
            )
            passed, rationale, revision = _parse_review(raw)
            reviews.append(
                ConstitutionalReview(
                    principle_id=principle["id"],
                    passed=passed,
                    rationale=rationale,
                    revision=revision if not passed else None,
                )
            )
            if not passed and revision:
                current = revision
                revised_any = True

        return ConstitutionalResult(
            original=draft, final=current, reviews=reviews, revised=revised_any
        )


def _parse_review(raw: str) -> tuple[bool, str, str | None]:
    passed = "PASSES: yes" in raw or "passes: yes" in raw.lower()
    rationale = ""
    revision: str | None = None
    if "RATIONALE:" in raw:
        rationale = raw.split("RATIONALE:", 1)[1].split("REVISION:", 1)[0].strip()
    if "REVISION:" in raw:
        rev = raw.split("REVISION:", 1)[1].strip()
        if rev:
            revision = rev
    return passed, rationale, revision
