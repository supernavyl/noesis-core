"""ReAct + Socratic + ToT reasoning smoke tests with stub LLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from noesis.reasoning.react import ReActLoop
from noesis.reasoning.tot import TreeOfThoughts


@dataclass
class _StubLLM:
    responses: list[str]

    def __post_init__(self) -> None:
        self._idx = 0

    async def complete(self, prompt: str, **_: object) -> str:
        out = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return out


def test_react_terminates_with_final_answer() -> None:
    async def _run() -> None:
        llm = _StubLLM(responses=["Thought: I know this.\nFINAL_ANSWER: 42"])
        loop = ReActLoop(llm=llm, tools={}, max_steps=3)
        trace = await loop.run("What is the answer?")
        assert trace.final_answer is not None
        assert "42" in trace.final_answer

    asyncio.run(_run())


def test_react_hits_max_steps() -> None:
    async def _run() -> None:
        llm = _StubLLM(responses=["Thought: I keep thinking.\nAction: nope\nAction Input: {}"])
        loop = ReActLoop(llm=llm, tools={}, max_steps=2)
        trace = await loop.run("Stuck question")
        assert trace.iterations == 2
        assert trace.final_answer is not None

    asyncio.run(_run())


def test_tot_explores_and_picks_best() -> None:
    async def _run() -> None:
        proposer = _StubLLM(responses=["a\nb\nc"])
        evaluator = _StubLLM(responses=["0.9", "0.5", "0.1"])
        tot = TreeOfThoughts(
            proposer=proposer, evaluator=evaluator, branching_factor=3, beam_width=1, max_depth=1
        )
        result = await tot.run("seed")
        assert result.explored >= 1
        assert result.best.score >= 0.0

    asyncio.run(_run())
