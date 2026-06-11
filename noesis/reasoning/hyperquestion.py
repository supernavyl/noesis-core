"""L2 — Socratic hyperquestioning.

After every ReAct step, the critic challenges the conclusion: "Why? Evidence? Counter? Edge case?"
The model must update or defend before continuing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class SocraticChallenge:
    claim: str
    questions: list[str]
    response: str
    revised: bool


@dataclass
class HyperquestionTrace:
    rounds: list[SocraticChallenge] = field(default_factory=list)
    final: str | None = None
    converged: bool = False


SOCRATIC_PROMPT = """You are a Socratic adversarial critic. The model has just claimed:

{claim}

Ask the 3 most penetrating questions that would force the claim to either be revised or strengthened.

Focus on:
- What evidence supports this? Is the evidence representative?
- What is the strongest counterargument?
- What edge case would break it?
- What assumption is hidden but load-bearing?

Output exactly 3 questions, one per line, no numbering or prefix.
"""

REVISE_PROMPT = """Critic asked:

{questions}

Reconsider your claim:

{claim}

Either:
(a) Revise the claim if any question successfully undermines it. Be specific about what changed and why.
(b) Defend the claim with the specific evidence each question demanded.

Output:
REVISED: <yes|no>
NEW_CLAIM: <revised or original claim>
JUSTIFICATION: <one paragraph>
"""


class SocraticLoop:
    def __init__(
        self,
        reasoner: LLMClient,
        critic: LLMClient,
        max_rounds: int = 3,
        convergence_threshold: int = 1,  # rounds without revision before halting
    ) -> None:
        self.reasoner = reasoner
        self.critic = critic
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold

    async def run(self, initial_claim: str) -> HyperquestionTrace:
        trace = HyperquestionTrace()
        claim = initial_claim
        stable_rounds = 0

        for _ in range(self.max_rounds):
            questions_raw = await self.critic.complete(SOCRATIC_PROMPT.format(claim=claim))
            questions = [q.strip() for q in questions_raw.splitlines() if q.strip()][:3]

            revised_raw = await self.reasoner.complete(
                REVISE_PROMPT.format(questions="\n".join(questions), claim=claim)
            )
            revised, new_claim = _parse_revision(revised_raw)

            trace.rounds.append(
                SocraticChallenge(
                    claim=claim, questions=questions, response=revised_raw, revised=revised
                )
            )

            if not revised:
                stable_rounds += 1
                if stable_rounds >= self.convergence_threshold:
                    trace.converged = True
                    trace.final = claim
                    return trace
            else:
                stable_rounds = 0
                claim = new_claim

        trace.final = claim
        return trace


def _parse_revision(raw: str) -> tuple[bool, str]:
    revised = "REVISED: yes" in raw.lower() or "revised: yes" in raw
    new_claim = raw
    for marker in ("NEW_CLAIM:", "new_claim:"):
        if marker in raw:
            new_claim = raw.split(marker, 1)[1].split("JUSTIFICATION:", 1)[0].strip()
            break
    return revised, new_claim
