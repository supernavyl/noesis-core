"""Query orchestrator — routes a query through model orchestra + reasoning depth selection.

Picks reasoning depth (L1-L5) based on query difficulty heuristics, then dispatches to the
right model(s) for that depth. End-to-end execution: ReAct → Socratic → ToT (optional) →
Constitutional review → audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

from noesis.core.model_registry import ModelRegistry, ModelSpec


class Depth(IntEnum):
    REACT = 1
    SOCRATIC = 2
    TREE_OF_THOUGHTS = 3
    HYPOTHETICAL = 4
    CONSTITUTIONAL = 5


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class QueryRequest:
    question: str
    session_id: str
    forced_depth: Depth | None = None
    forced_slot: str | None = None
    max_depth: Depth = Depth.CONSTITUTIONAL
    require_sandbox: bool = False
    meta: dict[str, Any] | None = None


@dataclass
class QueryResult:
    answer: str
    depth_used: Depth
    models_invoked: list[str]
    sources: list[dict[str, Any]] = field(default_factory=list)
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    audit_id: int | None = None


_CODE_MARKERS = ("```", "def ", "fn ", "class ", "import ", "function ", "->")


def _is_code_query(text: str) -> bool:
    return any(marker in text for marker in _CODE_MARKERS)


class Orchestrator:
    """Routes a query through the model orchestra at the appropriate reasoning depth.

    Components are injected so the same orchestrator can run with stub LLMs in tests and
    real vLLM clients in production.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        reasoner: LLMClient | None = None,
        critic: LLMClient | None = None,
        reviewer: LLMClient | None = None,
        tools: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self._reasoner = reasoner
        self._critic = critic
        self._reviewer = reviewer
        self._tools = tools

    # ── injection lazy-load ─────────────────────────────────────────────
    def _ensure_clients(self) -> tuple[LLMClient, LLMClient, LLMClient, dict[str, Any]]:
        if self._reasoner is None:
            from noesis.clients.llm import VLLMClient

            self._reasoner = VLLMClient(served_model="reasoner.primary")
        if self._critic is None:
            from noesis.clients.llm import VLLMClient

            self._critic = VLLMClient(served_model="critic.primary")
        if self._reviewer is None:
            self._reviewer = self._reasoner
        if self._tools is None:
            from noesis.reasoning.tools import default_tools

            self._tools = default_tools()
        return self._reasoner, self._critic, self._reviewer, self._tools

    # ── routing ─────────────────────────────────────────────────────────
    def pick_model(self, request: QueryRequest) -> ModelSpec:
        if request.forced_slot:
            slot, _, role = request.forced_slot.partition(".")
            return self.registry.get(slot, role or "primary")
        if len(request.question) > 50_000:
            return self.registry.get("synthesizer", "primary")
        if _is_code_query(request.question):
            return self.registry.get("coder", "primary")
        return self.registry.get("reasoner", "primary")

    def pick_depth(self, request: QueryRequest) -> Depth:
        if request.forced_depth:
            return min(request.forced_depth, request.max_depth)

        q = request.question.lower()
        depth = Depth.REACT
        if any(kw in q for kw in ("branches", "alternatives", "options for", "ways to")):
            depth = Depth.TREE_OF_THOUGHTS
        elif any(kw in q for kw in ("what if", "hypothesize", "imagine", "scenario")):
            depth = Depth.HYPOTHETICAL
        elif any(kw in q for kw in ("why", "prove", "compare", "tradeoff", "trade-off", "design")):
            depth = Depth.SOCRATIC
        return Depth(min(depth, request.max_depth))

    # ── execution ───────────────────────────────────────────────────────
    async def run(self, request: QueryRequest) -> QueryResult:
        from noesis.reasoning.constitutional import ConstitutionalReviewer
        from noesis.reasoning.hyperquestion import SocraticLoop
        from noesis.reasoning.react import ReActLoop
        from noesis.reasoning.tot import TreeOfThoughts

        reasoner, critic, reviewer, tools = self._ensure_clients()

        depth = self.pick_depth(request)
        model_spec = self.pick_model(request)
        models_invoked: list[str] = [model_spec.id]
        trace: list[dict[str, Any]] = []

        # L1 — ReAct (always runs)
        react = ReActLoop(llm=reasoner, tools=tools, max_steps=6)
        react_trace = await react.run(request.question)
        answer = react_trace.final_answer or "[no answer]"
        trace.append(
            {
                "phase": "react",
                "iterations": react_trace.iterations,
                "steps": [
                    {"thought": s.thought, "action": s.action, "observation": s.observation}
                    for s in react_trace.steps
                ],
            }
        )
        sources = _extract_sources_from_trace(react_trace)

        # L2 — Socratic (if requested)
        if depth >= Depth.SOCRATIC and request.max_depth >= Depth.SOCRATIC:
            socratic = SocraticLoop(reasoner=reasoner, critic=critic, max_rounds=2)
            socratic_trace = await socratic.run(answer)
            answer = socratic_trace.final or answer
            trace.append(
                {
                    "phase": "socratic",
                    "rounds": len(socratic_trace.rounds),
                    "converged": socratic_trace.converged,
                }
            )

        # L3 — Tree of Thoughts (if requested)
        if depth >= Depth.TREE_OF_THOUGHTS and request.max_depth >= Depth.TREE_OF_THOUGHTS:
            tot = TreeOfThoughts(
                proposer=reasoner, evaluator=critic, branching_factor=3, beam_width=2, max_depth=3
            )
            tot_result = await tot.run(request.question)
            # Append the best branch's full reasoning path as an observation
            best_path = " → ".join(tot_result.best.lineage())
            trace.append({"phase": "tot", "explored": tot_result.explored, "best": best_path[:500]})
            # Use ToT's best as additional context for the final answer
            answer = await _fuse_with_tot(reasoner, answer, best_path)

        # L5 — Constitutional review (always runs at max depth)
        if request.max_depth >= Depth.CONSTITUTIONAL:
            reviewer_wrapper = ConstitutionalReviewer(reviewer=reviewer)
            review = await reviewer_wrapper.review(answer)
            answer = review.final
            trace.append(
                {
                    "phase": "constitutional",
                    "revised": review.revised,
                    "principles_failed": [r.principle_id for r in review.reviews if not r.passed],
                }
            )

        return QueryResult(
            answer=answer,
            depth_used=depth,
            models_invoked=models_invoked,
            sources=sources,
            reasoning_trace=trace,
            confidence=_confidence_from_trace(trace),
        )


async def _fuse_with_tot(reasoner: LLMClient, current_answer: str, best_path: str) -> str:
    prompt = (
        "You produced this answer:\n\n"
        f"{current_answer}\n\n"
        "An alternative reasoning path explored:\n\n"
        f"{best_path}\n\n"
        "Integrate the strongest insights from both into a final, sharper answer. "
        "Output the final answer only."
    )
    return await reasoner.complete(prompt)


def _extract_sources_from_trace(react_trace: Any) -> list[dict[str, Any]]:
    """Pull URL/title from observations that came from search_corpus."""
    sources: list[dict[str, Any]] = []
    for step in getattr(react_trace, "steps", []):
        if not step.observation or step.action != "search_corpus":
            continue
        for line in step.observation.split("\n"):
            if line.strip().startswith("URL:"):
                sources.append({"url": line.split("URL:", 1)[1].strip()})
    return sources


def _confidence_from_trace(trace: list[dict[str, Any]]) -> float:
    # Heuristic: more phases run + Socratic converged + no constitutional revision → higher confidence
    confidence = 0.5
    for phase in trace:
        if phase["phase"] == "socratic" and phase.get("converged"):
            confidence += 0.15
        if phase["phase"] == "constitutional" and not phase.get("revised"):
            confidence += 0.15
        if phase["phase"] == "react" and phase.get("iterations", 0) >= 2:
            confidence += 0.1
    return min(1.0, confidence)
