"""Citation tracking — maps retrieved chunks to indices the model can cite as [n]."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    title: str
    url: str
    source: str

    def render(self) -> str:
        return f"[{self.index}] {self.source} — {self.title}\n    {self.url}"


@dataclass
class CitationTracker:
    _by_chunk: dict[str, Citation] = field(default_factory=dict)
    _order: list[Citation] = field(default_factory=list)

    def register(self, chunk_id: str, title: str, url: str, source: str) -> int:
        if chunk_id in self._by_chunk:
            return self._by_chunk[chunk_id].index
        index = len(self._order) + 1
        cite = Citation(index=index, chunk_id=chunk_id, title=title, url=url, source=source)
        self._by_chunk[chunk_id] = cite
        self._order.append(cite)
        return index

    def citations(self) -> list[Citation]:
        return list(self._order)

    def render_sources_block(self) -> str:
        if not self._order:
            return ""
        return "\n\n".join(c.render() for c in self._order)
