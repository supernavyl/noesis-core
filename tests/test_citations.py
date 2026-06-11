"""CitationTracker — chunk → index mapping + source-block rendering."""

from __future__ import annotations

from noesis.reasoning.citations import Citation, CitationTracker


def test_register_returns_consecutive_indices() -> None:
    tracker = CitationTracker()
    a = tracker.register(
        chunk_id="abc123",
        title="Flash Attention",
        url="https://arxiv.org/abs/2205.14135",
        source="arxiv",
    )
    b = tracker.register(
        chunk_id="def456", title="ReAct", url="https://arxiv.org/abs/2210.03629", source="arxiv"
    )
    assert a == 1
    assert b == 2


def test_register_deduplicates_same_chunk_id() -> None:
    tracker = CitationTracker()
    a = tracker.register(
        chunk_id="abc123",
        title="Flash Attention",
        url="https://arxiv.org/abs/2205.14135",
        source="arxiv",
    )
    b = tracker.register(
        chunk_id="abc123",
        title="Flash Attention",
        url="https://arxiv.org/abs/2205.14135",
        source="arxiv",
    )
    assert a == b == 1
    assert len(tracker.citations()) == 1


def test_render_sources_block_lists_each_once() -> None:
    tracker = CitationTracker()
    tracker.register(chunk_id="a", title="Paper A", url="https://example.com/a", source="arxiv")
    tracker.register(chunk_id="b", title="Paper B", url="https://example.com/b", source="github")
    rendered = tracker.render_sources_block()
    assert "[1] arxiv — Paper A" in rendered
    assert "https://example.com/a" in rendered
    assert "[2] github — Paper B" in rendered


def test_citations_returns_list_of_dataclasses() -> None:
    tracker = CitationTracker()
    tracker.register(chunk_id="a", title="t", url="u", source="s")
    cites = tracker.citations()
    assert len(cites) == 1
    assert isinstance(cites[0], Citation)
    assert cites[0].index == 1
    assert cites[0].chunk_id == "a"


def test_empty_tracker_renders_empty_block() -> None:
    tracker = CitationTracker()
    assert tracker.render_sources_block() == ""
