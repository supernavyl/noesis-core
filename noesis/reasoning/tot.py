"""L3 — Tree of Thoughts.

Generate N branches per step, score each, keep top-K. Yao et al. 2023.
Used when ReAct is uncertain (low confidence) or the problem space has multiple
viable paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import nlargest
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class ThoughtNode:
    text: str
    score: float
    parent: ThoughtNode | None = None
    depth: int = 0
    children: list[ThoughtNode] = field(default_factory=list)

    def lineage(self) -> list[str]:
        path: list[str] = []
        node: ThoughtNode | None = self
        while node is not None:
            path.append(node.text)
            node = node.parent
        return list(reversed(path))


@dataclass
class ToTResult:
    best: ThoughtNode
    explored: int
    max_depth_reached: int


class TreeOfThoughts:
    def __init__(
        self,
        proposer: LLMClient,
        evaluator: LLMClient,
        branching_factor: int = 3,
        beam_width: int = 2,
        max_depth: int = 4,
    ) -> None:
        self.proposer = proposer
        self.evaluator = evaluator
        self.branching = branching_factor
        self.beam = beam_width
        self.max_depth = max_depth

    async def run(self, question: str) -> ToTResult:
        root = ThoughtNode(text=question, score=1.0, depth=0)
        frontier: list[ThoughtNode] = [root]
        explored = 1
        max_depth = 0

        for depth in range(1, self.max_depth + 1):
            next_frontier: list[ThoughtNode] = []
            for node in frontier:
                proposals = await self._propose(node)
                scored = []
                for text in proposals:
                    score = await self._evaluate(text, node)
                    child = ThoughtNode(text=text, score=score, parent=node, depth=depth)
                    node.children.append(child)
                    scored.append(child)
                    explored += 1
                next_frontier.extend(scored)

            # Beam-keep top-k by score
            frontier = nlargest(self.beam, next_frontier, key=lambda n: n.score)
            max_depth = depth
            if not frontier:
                break

        best = max(_walk(root), key=lambda n: n.score)
        return ToTResult(best=best, explored=explored, max_depth_reached=max_depth)

    async def _propose(self, node: ThoughtNode) -> list[str]:
        prompt = (
            "Given this reasoning path:\n"
            + "\n".join(f"- {step}" for step in node.lineage())
            + f"\n\nPropose {self.branching} distinct next thoughts that advance the reasoning."
            f" One per line, no numbering."
        )
        raw = await self.proposer.complete(prompt)
        return [t.strip() for t in raw.splitlines() if t.strip()][: self.branching]

    async def _evaluate(self, text: str, parent: ThoughtNode) -> float:
        prompt = (
            f"Rate how promising this thought is on a 0.0-1.0 scale.\n"
            f"Parent: {parent.text}\nThought: {text}\n\n"
            f"Output only the number."
        )
        raw = await self.evaluator.complete(prompt)
        try:
            return max(0.0, min(1.0, float(raw.strip().splitlines()[0])))
        except (ValueError, IndexError):
            return 0.5


def _walk(node: ThoughtNode) -> list[ThoughtNode]:
    nodes = [node]
    for child in node.children:
        nodes.extend(_walk(child))
    return nodes
