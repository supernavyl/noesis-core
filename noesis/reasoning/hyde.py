"""HyDE — Hypothetical Document Embeddings.

Gao et al. 2022. Generate a plausible answer to the query, embed THAT for retrieval.
Embeddings of answer-shaped text sit closer to corpus passages than question-shaped text,
so recall improves on synthesis queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


_HYDE_PROMPT = """Write a single concise paragraph that PLAUSIBLY answers the following question.

Do not say "I don't know" or qualify. Write as if you are an expert. Do not preface or summarize.
Just write the paragraph. The paragraph will be used as a search query.

Question: {q}

Paragraph:"""


@dataclass
class HypotheticalExpander:
    llm: LLMClient
    include_query: bool = True
    max_tokens: int = 384

    async def expand(self, query: str) -> str:
        hypothesis = await self.llm.complete(
            _HYDE_PROMPT.format(q=query), max_tokens=self.max_tokens, temperature=0.4
        )
        hypothesis = _strip_reasoning_wrapper(hypothesis).strip()
        if not self.include_query:
            return hypothesis
        return f"{query}\n\n{hypothesis}"


def _strip_reasoning_wrapper(text: str) -> str:
    """If the LLM returned <reasoning>…</reasoning>, unwrap it.

    See noesis/clients/llm.py chat_text fallback — Qwen3.5/R1/Phi-4-reasoning sometimes return
    only the reasoning field, which our client wraps in tags.
    """
    if "<reasoning>" in text and "</reasoning>" in text:
        start = text.index("<reasoning>") + len("<reasoning>")
        end = text.index("</reasoning>")
        return text[start:end].strip()
    return text
