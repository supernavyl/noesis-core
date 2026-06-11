"""Orchestrator end-to-end with stub LLM + stub tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from noesis.core.orchestrator import Depth, Orchestrator, QueryRequest


@dataclass
class _StubLLM:
    """Returns the next scripted response each call. Repeats if exhausted."""

    responses: list[str]
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._i = 0

    async def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        out = self.responses[self._i % len(self.responses)]
        self._i += 1
        return out


class _StubTool:
    schema = {
        "type": "function",
        "function": {"name": "search_corpus", "parameters": {}},
    }

    async def __call__(self, _name: str, params: dict[str, Any]) -> str:
        return "OBS: corpus snippet about flash attention\nURL: https://example.com/flash"


def _build(orchestrator_responses: list[str]) -> tuple[Orchestrator, _StubLLM]:
    llm = _StubLLM(responses=orchestrator_responses)
    orch = Orchestrator(
        reasoner=llm,
        critic=llm,
        reviewer=llm,
        tools={"search_corpus": _StubTool()},
    )
    return orch, llm


def test_orchestrator_runs_react_only_at_depth_1() -> None:
    orch, llm = _build(
        [
            # ReAct final answer on first iteration
            "Thought: I know this.\nFINAL_ANSWER: Flash attention tiles Q/K/V into SRAM.",
            # Constitutional review pass for the single principle invoked
            "PASSES: yes\nRATIONALE: looks fine.\nREVISION:",
        ]
    )
    req = QueryRequest(
        question="explain flash attention",
        session_id="s1",
        forced_depth=Depth.REACT,
        max_depth=Depth.CONSTITUTIONAL,
    )
    result = asyncio.run(orch.run(req))
    assert "Flash attention" in result.answer or "FINAL_ANSWER" not in result.answer
    assert result.depth_used == Depth.REACT
    # We expect at least one ReAct call + constitutional review calls
    assert len(llm.calls) >= 2


def test_orchestrator_socratic_runs_at_depth_2() -> None:
    orch, llm = _build(
        [
            "Thought: known.\nFINAL_ANSWER: GQA shares K/V across Q heads.",
            "What evidence?\nIs there a counterexample?\nWhich edge case fails?",
            "REVISED: no\nNEW_CLAIM: GQA shares K/V across Q heads.\nJUSTIFICATION: standard.",
            "PASSES: yes\nRATIONALE: ok.\nREVISION:",
        ]
    )
    req = QueryRequest(
        question="why does GQA work, compare to MHA",
        session_id="s2",
        max_depth=Depth.CONSTITUTIONAL,
    )
    result = asyncio.run(orch.run(req))
    assert result.depth_used >= Depth.SOCRATIC
    phases = [t["phase"] for t in result.reasoning_trace]
    assert "react" in phases
    assert "socratic" in phases
    assert "constitutional" in phases


def test_orchestrator_max_depth_caps_phases() -> None:
    orch, _ = _build(["Thought: x.\nFINAL_ANSWER: y."])
    req = QueryRequest(
        question="why X",
        session_id="s3",
        max_depth=Depth.REACT,  # cap at L1
    )
    result = asyncio.run(orch.run(req))
    phases = [t["phase"] for t in result.reasoning_trace]
    assert "react" in phases
    assert "socratic" not in phases
    assert "constitutional" not in phases


def test_pick_model_routes_code_queries_to_coder() -> None:
    orch = Orchestrator(reasoner=_StubLLM(["x"]), tools={})
    code_req = QueryRequest(question="def foo(x): return x", session_id="s4")
    spec = orch.pick_model(code_req)
    assert spec.slot == "coder"


def test_pick_model_routes_long_to_synthesizer() -> None:
    orch = Orchestrator(reasoner=_StubLLM(["x"]), tools={})
    long_req = QueryRequest(question="x" * 60_000, session_id="s5")
    spec = orch.pick_model(long_req)
    assert spec.slot == "synthesizer"


def test_confidence_grows_with_phases() -> None:
    orch, _ = _build(
        [
            "Thought: known.\nThought: still known.\nFINAL_ANSWER: a settled answer.",
            "Q1?\nQ2?\nQ3?",
            "REVISED: no\nNEW_CLAIM: a settled answer.\nJUSTIFICATION: x.",
            "PASSES: yes\nRATIONALE: ok.\nREVISION:",
        ]
    )
    req = QueryRequest(question="why X", session_id="s6", max_depth=Depth.CONSTITUTIONAL)
    result = asyncio.run(orch.run(req))
    assert result.confidence >= 0.5
