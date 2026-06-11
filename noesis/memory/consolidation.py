"""Memory consolidation — episodic → semantic.

Runs during the dream cycle's NREM phase. Reads recent episodic memory, extracts durable
facts via the synthesizer model, and writes them to the semantic store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from noesis.memory.episodic import EpisodicStore
from noesis.memory.semantic import SemanticFact, SemanticStore


class LLMClient(Protocol):
    async def complete(self, prompt: str, **kw: Any) -> str: ...


EXTRACTION_PROMPT = """Extract durable facts from the following conversation history. A "durable fact"
is something that should remain true beyond the immediate conversation — a concept defined, a
relationship established, a technique learned.

For each fact, output one line in the format:
FACT: <statement> | CONF: <0.0-1.0> | SRC: <episodic_id>

Reject anything ephemeral, opinionated, or personal-context. Maximum 10 facts.

History:
{history}
"""


@dataclass
class ConsolidationReport:
    episodes_reviewed: int
    facts_extracted: int
    facts_written: int


class Consolidator:
    def __init__(
        self,
        synthesizer: LLMClient,
        episodic: EpisodicStore | None = None,
        semantic: SemanticStore | None = None,
        min_confidence: float = 0.75,
    ) -> None:
        self.synthesizer = synthesizer
        self.episodic = episodic or EpisodicStore()
        self.semantic = semantic or SemanticStore()
        self.min_confidence = min_confidence

    async def run(self, window_days: int = 7) -> ConsolidationReport:
        episodes = await self.episodic.recent_window(days=window_days)
        if not episodes:
            return ConsolidationReport(episodes_reviewed=0, facts_extracted=0, facts_written=0)

        history = "\n".join(f"[id={e['id']}] {e['role']}: {e['content'][:1000]}" for e in episodes)
        raw = await self.synthesizer.complete(EXTRACTION_PROMPT.format(history=history))

        facts = _parse_facts(raw)
        written = 0
        for fact in facts:
            if fact.confidence >= self.min_confidence:
                await self.semantic.write(fact)
                written += 1

        return ConsolidationReport(
            episodes_reviewed=len(episodes),
            facts_extracted=len(facts),
            facts_written=written,
        )


def _parse_facts(raw: str) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    for line in raw.splitlines():
        if not line.startswith("FACT:"):
            continue
        parts = [p.strip() for p in line[5:].split("|")]
        if len(parts) < 3:
            continue
        statement = parts[0]
        try:
            conf = float(parts[1].replace("CONF:", "").strip())
        except ValueError:
            conf = 0.5
        src = parts[2].replace("SRC:", "").strip()
        facts.append(SemanticFact(statement=statement, confidence=conf, provenance=[src]))
    return facts
