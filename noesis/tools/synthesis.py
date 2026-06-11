"""Tool synthesis — model designs new tools when existing ones don't fit.

Pipeline: problem → spec → implementation (Python) → sandbox-verified tests → registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from noesis.sandbox.executor import DockerExecutor


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class SynthesizedTool:
    name: str
    description: str
    signature: dict[str, Any]
    implementation: str
    tests: list[str]
    verified: bool = False


DESIGN_PROMPT = """A new tool is needed. Problem:

{problem}

Design a Python tool. Output exactly:

NAME: <snake_case_name>
DESCRIPTION: <one sentence>
SIGNATURE: <JSON schema of input parameters>
IMPLEMENTATION:
```python
async def tool(params: dict) -> Any:
    # implementation
    ...
```
TESTS:
```python
# 3 self-checking tests that print PASS / FAIL
```
"""


class ToolSynthesizer:
    def __init__(self, coder: LLMClient, executor: DockerExecutor | None = None) -> None:
        self.coder = coder
        self.executor = executor or DockerExecutor()

    async def synthesize(self, problem: str) -> SynthesizedTool | None:
        raw = await self.coder.complete(DESIGN_PROMPT.format(problem=problem))
        tool = _parse(raw)
        if not tool:
            return None
        verified = await self._verify(tool)
        tool.verified = verified
        return tool

    async def _verify(self, tool: SynthesizedTool) -> bool:
        # Concatenate implementation + tests and run in sandbox.
        program = tool.implementation + "\n\n" + "\n".join(tool.tests)
        result = await self.executor.run(program, lang="python", timeout_s=30)
        return "PASS" in result.stdout and "FAIL" not in result.stdout


def _parse(raw: str) -> SynthesizedTool | None:
    try:
        import orjson

        name = raw.split("NAME:", 1)[1].split("\n", 1)[0].strip()
        desc = raw.split("DESCRIPTION:", 1)[1].split("\n", 1)[0].strip()
        sig_str = raw.split("SIGNATURE:", 1)[1].split("IMPLEMENTATION:", 1)[0].strip()
        sig = orjson.loads(sig_str)
        impl = raw.split("IMPLEMENTATION:", 1)[1].split("TESTS:", 1)[0]
        impl = _extract_code_block(impl)
        tests_block = raw.split("TESTS:", 1)[1]
        tests = [_extract_code_block(tests_block)]
        return SynthesizedTool(
            name=name,
            description=desc,
            signature=sig,
            implementation=impl,
            tests=tests,
        )
    except (IndexError, ValueError, KeyError):
        return None


def _extract_code_block(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    # parts: [pre, "python\n...", post]
    if len(parts) < 2:
        return text.strip()
    body = parts[1]
    if body.startswith("python"):
        body = body[len("python") :]
    return body.strip()
