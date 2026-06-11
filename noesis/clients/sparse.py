"""SPLADE-v3 sparse encoder — runs in-process via transformers.

Outputs a sparse vector keyed by token id. Used alongside dense embeddings for hybrid retrieval.
Lightweight enough to run on CPU; will move to GPU later if needed.

Fail-open: if the model can't load (gated HF repo, network error, missing token), encode()
returns an empty dict. The pipeline degrades to dense + BM25 hybrid and keeps running. A
single warning is logged the first time load fails; subsequent calls are silent.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # avoid import cost on plain config tests
    import torch  # noqa: F401  (intentionally present for type-checkers)
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: F401


# SPLADE-v3 (naver/splade-v3) is gated. The ungated alternative
# (prithivida/Splade_PP_en_v1) loads but vector_store.search() uses an outdated
# dict-based qdrant prefetch format that 400s on qdrant_client 1.17 when sparse
# vectors are populated. Until vector_store.search() is fixed to use
# qdrant_client.models.Prefetch, keep SPLADE pointed at the gated repo so it
# fail-opens — preserving the existing dense+BM25-only retrieval path.
_MODEL_NAME = "naver/splade-v3"


class SpladeEncoder:
    """Async wrapper around the SPLADE masked-LM encoder. Fail-open on load errors."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None
        self._init_lock = asyncio.Lock()
        self._disabled = False  # set on first failed load
        self._load_error: str | None = None

    async def _ensure_loaded(self) -> bool:
        """Return True if the model is loaded and usable, False if disabled."""
        if self._model is not None:
            return True
        if self._disabled:
            return False
        async with self._init_lock:
            if self._model is not None:
                return True
            if self._disabled:
                return False
            try:
                import torch  # noqa: F401  (imported here so import errors fail-open)
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
                self._model.eval()
                return True
            except Exception as exc:  # gated repo, no network, missing token, etc.
                self._disabled = True
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "SPLADE model '{}' failed to load — sparse retrieval disabled. "
                    "Pipeline continues with dense + BM25 only. "
                    "Cause: {}. To enable: accept the gated repo at "
                    "https://huggingface.co/{} and set HF_TOKEN, or switch "
                    "SpladeEncoder(model_name=...) to an ungated model.",
                    self.model_name,
                    self._load_error,
                    self.model_name,
                )
                return False

    async def encode(self, text: str) -> dict[int, float]:
        if not await self._ensure_loaded():
            return {}
        return await asyncio.to_thread(self._encode_sync, text)

    async def encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        if not await self._ensure_loaded():
            return [{} for _ in texts]
        return await asyncio.to_thread(self._encode_batch_sync, texts)

    def _encode_sync(self, text: str) -> dict[int, float]:
        return self._encode_batch_sync([text])[0]

    def _encode_batch_sync(self, texts: list[str]) -> list[dict[int, float]]:
        import torch

        assert self._tok is not None and self._model is not None

        with torch.no_grad():
            enc = self._tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc).logits  # [B, L, V]
            # SPLADE pooling: log(1 + ReLU(logits)) then max over sequence
            x = torch.log1p(torch.relu(out)) * enc["attention_mask"].unsqueeze(-1)
            pooled, _ = x.max(dim=1)  # [B, V]

        results: list[dict[int, float]] = []
        for row in pooled:
            nonzero = row.nonzero(as_tuple=False).squeeze(-1)
            results.append({int(idx): float(row[idx]) for idx in nonzero})
        return results
