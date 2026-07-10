"""Chunking — semantic boundary chunking with parent-child hierarchy.

Strategy:
1. Split into sentences via a fast tokenizer-aware splitter.
2. Embed each sentence with the embedder.
3. Place chunk boundaries where adjacent-sentence cosine similarity drops below a threshold.
4. Build a parent-chunk index for retrieval (small chunks for matching, larger parents for context).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    parent_id: str | None = None
    meta: dict[str, Any] | None = None


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    rough = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in rough if s.strip()]


def fixed_chunks(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        piece = text[i : i + size]
        if not piece.strip():
            continue
        chunks.append(Chunk(text=piece, start=i, end=i + len(piece)))
    return chunks


def semantic_chunks(
    sentences: list[str],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.65,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 2000,
) -> list[Chunk]:
    """Group sentences into chunks at low-similarity boundaries."""
    if not sentences:
        return []
    chunks: list[Chunk] = []
    buf: list[str] = []
    cursor = 0
    buf_start = 0

    for i, sent in enumerate(sentences):
        buf.append(sent)
        cursor += len(sent) + 1

        next_sim = _cosine(embeddings[i], embeddings[i + 1]) if i + 1 < len(sentences) else 0.0
        flush = (
            i + 1 == len(sentences)
            or (next_sim < similarity_threshold and sum(len(s) for s in buf) >= min_chunk_chars)
            or sum(len(s) for s in buf) >= max_chunk_chars
        )
        if flush:
            text = " ".join(buf).strip()
            chunks.append(Chunk(text=text, start=buf_start, end=buf_start + len(text)))
            buf = []
            buf_start = cursor

    return chunks


def build_parent_chunks(child_chunks: list[Chunk], group_size: int = 4) -> list[Chunk]:
    """Group N child chunks into one parent chunk for context expansion at retrieval time."""
    parents: list[Chunk] = []
    for i in range(0, len(child_chunks), group_size):
        group = child_chunks[i : i + group_size]
        if not group:
            continue
        text = " ".join(c.text for c in group)
        parents.append(Chunk(text=text, start=group[0].start, end=group[-1].end))
    return parents


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-12)
