"""Smoke tests for config loading."""

from __future__ import annotations

from noesis.core.config import (
    constitution,
    dream_config,
    ingestion_config,
    models_config,
    settings,
)


def test_models_config_loads() -> None:
    cfg = models_config()
    assert "reasoner" in cfg
    # Reasoner primary must be abliterated. Accept either hyphen (HF repo) or
    # underscore (Ollama tag) namespacing.
    primary_id = cfg["reasoner"]["primary"]["id"]
    assert "huihui" in primary_id.lower() and "abliterated" in primary_id.lower()


def test_ingestion_config_loads() -> None:
    cfg = ingestion_config()
    assert "arxiv" in cfg["sources"]
    assert cfg["sources"]["arxiv"]["enabled"] is True


def test_dream_config_loads() -> None:
    cfg = dream_config()
    assert "phases" in cfg
    assert cfg["phases"]["train"]["method"] == "lora"


def test_constitution_loads_and_is_immutable() -> None:
    c = constitution()
    assert c.modifiable_by_model is False
    assert any(p["id"] == "radical_honesty" for p in c.principles)
    assert any(p["id"] == "zero_fabrication" for p in c.principles)


def test_settings_postgres_dsn() -> None:
    s = settings()
    assert "postgresql://" in s.postgres_dsn
