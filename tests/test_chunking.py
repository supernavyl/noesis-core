"""Chunking smoke tests — pure-Python, no services."""

from __future__ import annotations

from noesis.knowledge.chunking import (
    build_parent_chunks,
    fixed_chunks,
    semantic_chunks,
    split_sentences,
)


def test_split_sentences_basic() -> None:
    text = "First sentence. Second sentence! Third one? Fourth."
    sents = split_sentences(text)
    assert len(sents) >= 3


def test_fixed_chunks_respects_size() -> None:
    text = "a" * 1500
    chunks = fixed_chunks(text, size=500, overlap=50)
    assert all(len(c.text) <= 500 for c in chunks)
    assert len(chunks) >= 2


def test_semantic_chunks_boundary() -> None:
    sentences = [
        "AI is a field of computer science.",
        "It studies intelligent agents.",
        "Bananas are yellow.",
    ]
    embs = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]
    chunks = semantic_chunks(sentences, embs, similarity_threshold=0.5, min_chunk_chars=10)
    assert len(chunks) >= 1


def test_parent_chunks_group() -> None:
    from noesis.knowledge.chunking import Chunk

    children = [Chunk(text=str(i) * 50, start=i * 50, end=(i + 1) * 50) for i in range(10)]
    parents = build_parent_chunks(children, group_size=4)
    assert len(parents) == 3
