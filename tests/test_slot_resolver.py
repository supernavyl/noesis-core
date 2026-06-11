"""resolve_slot — model ID lookup honors backend-specific alt_ids."""

from __future__ import annotations

from noesis.clients.llm import resolve_slot


def test_resolve_default_returns_primary_id() -> None:
    resolved = resolve_slot("reasoner.primary", backend="ollama")
    # Active config picks the Ollama-native tag for the reasoner.
    assert "huihui" in resolved.lower()


def test_resolve_alt_id_for_vllm_backend() -> None:
    resolved = resolve_slot("reasoner.primary", backend="vllm")
    # vLLM should get the HF repo id, not the Ollama tag.
    assert "/" in resolved
    assert ":" not in resolved


def test_resolve_unknown_slot_returns_input() -> None:
    out = resolve_slot("does_not_exist.primary", backend="ollama")
    assert out == "does_not_exist.primary"


def test_resolve_coder_slot() -> None:
    resolved = resolve_slot("coder.primary", backend="ollama")
    assert "coder" in resolved.lower()


def test_resolve_embedder_alt_for_tei() -> None:
    resolved = resolve_slot("embedder.primary", backend="tei")
    # TEI backend should resolve to the HF repo, not the Ollama tag.
    assert "Qwen" in resolved or "/" in resolved
