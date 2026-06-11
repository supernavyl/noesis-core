"""L4 — Hypothetical engine.

Given a claim or proposed answer, generate counterfactual / what-if variants, design empirical
tests for each, run them in the sandbox, and report which hold up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from noesis.sandbox.executor import DockerExecutor, ExecResult


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class Hypothesis:
    claim: str
    test_code: str
    test_language: str = "python"
    expected: str | None = None


@dataclass
class HypothesisRun:
    hypothesis: Hypothesis
    exec_result: ExecResult
    supports_claim: bool


@dataclass
class HypothesisReport:
    original_claim: str
    runs: list[HypothesisRun] = field(default_factory=list)
    surviving: list[Hypothesis] = field(default_factory=list)
    refuted: list[Hypothesis] = field(default_factory=list)


DESIGN_PROMPT = """You are an empirical scientist. The model claims:

{claim}

Design 3 distinct executable tests that would falsify this claim if it is false. Each test must
be a self-contained snippet that prints 'PASS' on success or 'FAIL' (with diagnostic) otherwise.

Output as 3 blocks, each:
TEST <i>:
LANG: <python|bash|node>
CODE:
<code here>
EXPECTED: <PASS|FAIL>
"""


class HypothesisEngine:
    def __init__(self, designer: LLMClient, executor: DockerExecutor | None = None) -> None:
        self.designer = designer
        self.executor = executor or DockerExecutor()

    async def run(self, claim: str) -> HypothesisReport:
        raw = await self.designer.complete(DESIGN_PROMPT.format(claim=claim))
        hypotheses = _parse_tests(raw)
        report = HypothesisReport(original_claim=claim)

        for h in hypotheses:
            exec_result = await self.executor.run(h.test_code, lang=h.test_language)  # type: ignore[arg-type]
            supports = "PASS" in exec_result.stdout and exec_result.exit_code == 0
            report.runs.append(
                HypothesisRun(hypothesis=h, exec_result=exec_result, supports_claim=supports)
            )
            (report.surviving if supports else report.refuted).append(h)

        return report


def _parse_tests(raw: str) -> list[Hypothesis]:
    tests: list[Hypothesis] = []
    blocks = raw.split("TEST ")
    for block in blocks[1:]:
        try:
            lang_line = next(ln for ln in block.splitlines() if ln.strip().startswith("LANG:"))
            lang = lang_line.split(":", 1)[1].strip()
            code_start = block.index("CODE:") + len("CODE:")
            expected_start = block.index("EXPECTED:")
            code = block[code_start:expected_start].strip()
            expected = block[expected_start + len("EXPECTED:") :].strip().splitlines()[0]
            tests.append(
                Hypothesis(claim="<derived>", test_code=code, test_language=lang, expected=expected)
            )
        except (StopIteration, ValueError):
            continue
    return tests
