"""HypotheticalExpander — drafts an answer for HyDE retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from noesis.reasoning.hyde import HypotheticalExpander


@dataclass
class _StubLLM:
    responses: list[str]
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._i = 0

    async def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        out = self.responses[self._i % len(self.responses)]
        self._i += 1
        return out


def test_expand_returns_hypothetical_answer() -> None:
    async def _run() -> None:
        llm = _StubLLM(
            responses=[
                "Flash Attention reduces O(N^2) HBM reads by tiling Q, K, V into SRAM blocks."
            ]
        )
        hyde = HypotheticalExpander(llm=llm)
        result = await hyde.expand("What does Flash Attention reduce?")
        assert "Flash Attention" in result
        assert "tiling" in result
        assert len(llm.calls) == 1

    asyncio.run(_run())


def test_expand_combines_query_and_hypothesis_for_embedding() -> None:
    async def _run() -> None:
        llm = _StubLLM(responses=["hypothetical paragraph about FA"])
        hyde = HypotheticalExpander(llm=llm, include_query=True)
        result = await hyde.expand("flash attention?")
        assert "flash attention?" in result
        assert "hypothetical paragraph about FA" in result

    asyncio.run(_run())


def test_expand_with_include_query_false_returns_hypothesis_only() -> None:
    async def _run() -> None:
        llm = _StubLLM(responses=["only the hypothesis"])
        hyde = HypotheticalExpander(llm=llm, include_query=False)
        result = await hyde.expand("ignore me")
        assert result == "only the hypothesis"
        assert "ignore me" not in result

    asyncio.run(_run())


def test_expand_strips_reasoning_wrapper() -> None:
    """Reasoning models (Qwen3.5, R1) sometimes return <reasoning>…</reasoning>.
    HyDE wants the prose, not the chain-of-thought."""

    async def _run() -> None:
        wrapped = "<reasoning>\nthe model thought a lot\n</reasoning>"
        llm = _StubLLM(responses=[wrapped])
        hyde = HypotheticalExpander(llm=llm, include_query=False)
        result = await hyde.expand("q")
        # Fallback to the reasoning text when content was empty
        assert "the model thought a lot" in result

    asyncio.run(_run())
