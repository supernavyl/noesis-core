"""Local workspace ingestion source.

Walks project directories and ingests code, docs, ADRs, memory files, and
architectural decisions into the NOESIS corpus. Each file becomes one
IngestionItem; chunking is handled downstream by the pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource

# File extensions to ingest per language/doc type
_TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".svelte",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".bash",
}

# Directories to always skip
_SKIP_DIRS = {
    "target",
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".cache",
    "coverage",
    ".tox",
    "eggs",
    ".eggs",
    "htmlcov",
    "emily-env",
    "site-packages",
    ".cargo",
}

# Files to always skip
_SKIP_FILES = {
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "uv.lock",
    ".DS_Store",
    "thumbs.db",
}

# Max file size (bytes) — skip huge generated files
_MAX_FILE_BYTES = 200_000


class WorkspaceSource(IngestionSource):
    """Ingest local project directories into NOESIS."""

    name = "workspace"

    def __init__(
        self,
        roots: list[str | Path],
        label: str = "workspace",
        extensions: set[str] | None = None,
        max_file_bytes: int = _MAX_FILE_BYTES,
    ) -> None:
        self.roots = [Path(r).expanduser().resolve() for r in roots]
        self.label = label
        self.extensions = extensions or _TEXT_EXTENSIONS
        self.max_file_bytes = max_file_bytes

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        for root in self.roots:
            if not root.exists():
                logger.warning("workspace root does not exist: {}", root)
                continue
            logger.info("scanning workspace root: {}", root)
            async for item in self._walk(root):
                yield item

    async def _walk(self, root: Path) -> AsyncIterator[IngestionItem]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            # Skip hidden dirs and blacklisted dirs
            parts = path.relative_to(root).parts
            if any(p.startswith(".") or p in _SKIP_DIRS for p in parts[:-1]):
                continue
            if path.name in _SKIP_FILES:
                continue
            if path.suffix.lower() not in self.extensions:
                continue
            if path.stat().st_size > self.max_file_bytes:
                logger.debug("skipping large file: {} ({} bytes)", path, path.stat().st_size)
                continue

            item = self._build_item(path, root)
            if item:
                yield item

    def _build_item(self, path: Path, root: Path) -> IngestionItem | None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            logger.debug("cannot read {}: {}", path, exc)
            return None

        if not text or len(text) < 30:
            return None

        rel = path.relative_to(root)
        source_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()

        # Derive a human title from filename + parent path
        stem = path.stem.replace("_", " ").replace("-", " ")
        parent = str(rel.parent) if str(rel.parent) != "." else ""
        title = f"{parent}/{stem}".lstrip("/") if parent else stem

        tags = ["workspace", self.label, path.suffix.lstrip(".")]
        if any(kw in path.name.lower() for kw in ("adr", "decision", "arch")):
            tags.append("architecture")
        if any(kw in path.name.lower() for kw in ("readme", "claude", "memory")):
            tags.append("project-docs")

        return IngestionItem(
            source=f"workspace:{self.label}",
            source_id=source_id,
            url=str(path),
            title=title,
            text=f"# {title}\n\nFile: {path}\n\n{text}",
            published_at=mtime,
            tags=tags,
        )
