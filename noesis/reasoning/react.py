"""L1 — ReAct loop (think → act → observe).

Foundation of every higher-depth reasoning mode. Yao et al. 2022.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


@dataclass
class ReActStep:
    thought: str
    action: str | None
    action_input: dict[str, Any] | None
    observation: str | None


@dataclass
class ReActTrace:
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str | None = None
    iterations: int = 0


ToolFn = Callable[[str, dict[str, Any]], Awaitable[str]]


class ReActLoop:
    """ReAct executor with configurable tool registry and step ceiling."""

    def __init__(
        self,
        llm: LLMClient,
        tools: dict[str, ToolFn],
        max_steps: int = 8,
        stop_token: str = "FINAL_ANSWER:",
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.stop_token = stop_token

    async def run(self, question: str) -> ReActTrace:
        trace = ReActTrace()
        prompt = self._initial_prompt(question)

        for _ in range(self.max_steps):
            response = await self.llm.complete(prompt)
            trace.iterations += 1

            if self.stop_token in response:
                trace.final_answer = response.split(self.stop_token, 1)[1].strip()
                return trace

            step = self._parse_step(response)
            trace.steps.append(step)

            if step.action and step.action in self.tools and step.action_input is not None:
                observation = await self.tools[step.action](step.action, step.action_input)
                step.observation = observation
                prompt = self._append_observation(prompt, response, observation)
            else:
                prompt = prompt + response

        trace.final_answer = "[max steps reached without final answer]"
        return trace

    # Prompt templating — keep minimal here, gets compiled via DSPy in P2 proper.
    def _initial_prompt(self, question: str) -> str:
        tools_doc = "\n".join(f"- {name}" for name in self.tools)
        return (
            f"Question: {question}\n\n"
            f"Available tools:\n{tools_doc}\n\n"
            f"Use the format:\n"
            f"Thought: <reasoning>\n"
            f"Action: <tool name>\n"
            f"Action Input: <JSON>\n"
            f"Observation: <result>\n"
            f"... (repeat as needed) ...\n"
            f"{self.stop_token} <final answer>\n\n"
            f"Thought:"
        )

    def _parse_step(self, raw: str) -> ReActStep:
        # Minimal parser — full grammar comes in P2.
        thought = _extract_after(raw, "Thought:") or raw.strip()
        action = _extract_after(raw, "Action:")
        action_input_raw = _extract_after(raw, "Action Input:")

        action_input: dict[str, Any] | None = None
        if action_input_raw:
            try:
                import orjson

                action_input = orjson.loads(action_input_raw)
            except Exception:
                action_input = {"raw": action_input_raw}

        return ReActStep(
            thought=thought, action=action, action_input=action_input, observation=None
        )

    def _append_observation(self, prompt: str, response: str, observation: str) -> str:
        return f"{prompt}{response}\nObservation: {observation}\nThought:"


def _extract_after(text: str, marker: str) -> str | None:
    if marker not in text:
        return None
    after = text.split(marker, 1)[1]
    next_marker = min(
        (
            after.find(m)
            for m in ("Thought:", "Action:", "Action Input:", "Observation:", "FINAL_ANSWER:")
            if after.find(m) != -1
        ),
        default=-1,
    )
    return after[:next_marker].strip() if next_marker != -1 else after.strip()
