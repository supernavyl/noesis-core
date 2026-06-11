"""LLM client — talks to any OpenAI-compatible inference server.

Works with vLLM, Ollama (default), TGI, llama.cpp server. Provides `.complete()` for raw
text completion and `.chat()` for message-list inputs. Handles retries + streaming.

Model resolution: `slot=` (e.g. "reasoner.primary") looks the actual model id up from
`configs/models.yaml`, using the active backend's id where alt_ids are defined.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from noesis.core.config import models_config, settings


@dataclass
class Message:
    role: str
    content: str


def resolve_slot(slot_path: str, backend: str | None = None) -> str:
    """Look up the model id for a slot path like 'reasoner.primary' or 'coder.primary'.

    If the slot defines alt_ids for the active backend, return that. Else return the default id.
    """
    backend = backend or settings().backend
    slot, _, role = slot_path.partition(".")
    role = role or "primary"
    try:
        spec = models_config()[slot][role]
        alt = spec.get("alt_ids", {})
        if backend in alt:
            return alt[backend]
        return spec["id"]
    except (KeyError, TypeError):
        return slot_path


class VLLMClient:
    """Async client targeting any OpenAI-compatible inference server (vLLM, Ollama, TGI, …)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        served_model: str = "reasoner.primary",
        timeout_s: float = 600.0,
        temperature: float = 0.7,
        max_tokens: int = 4096,  # reasoning models need headroom
        backend: str | None = None,
    ) -> None:
        s = settings()
        self.backend = backend or s.backend
        self.base_url = (base_url or s.vllm_base_url).rstrip("/")
        self.api_key = api_key or s.vllm_api_key
        # If served_model looks like a slot path, resolve it. Else use as-is.
        if "." in served_model and "/" not in served_model and ":" not in served_model:
            self.served_model = resolve_slot(served_model, backend=self.backend)
        else:
            self.served_model = served_model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> VLLMClient:
        self._http = httpx.AsyncClient(timeout=self.timeout_s)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
        return self._http

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        """Text completion. Ollama's /v1/completions is mediocre for instruct/reasoning models —
        we route through /v1/chat/completions and concatenate reasoning + content so callers
        (e.g. ReAct loop) get parseable output regardless of whether the model thinks first.
        """
        # Route via chat for reasoning-model compatibility.
        text = await self.chat_text(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def chat(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        http = await self._ensure()
        normalized = [
            (m if isinstance(m, dict) else {"role": m.role, "content": m.content}) for m in messages
        ]
        body: dict[str, Any] = {
            "model": model or self.served_model,
            "messages": normalized,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        resp = await http.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=body
        )
        resp.raise_for_status()
        return resp.json()

    async def chat_text(self, messages: list[Message] | list[dict[str, str]], **kw: Any) -> str:
        """Return the assistant text.

        Reasoning models on Ollama (Qwen3.5, DeepSeek-R1, Phi-4-reasoning) split output into
        `reasoning` + `content`. We prefer `content`. If the model ran out of budget mid-think
        (`content` empty, `reasoning` populated, finish=length), we wrap reasoning as the
        answer so the caller still gets parseable text — and the orchestrator can detect it.
        """
        result = await self.chat(messages, **kw)
        msg = result["choices"][0]["message"]
        content = msg.get("content") or ""
        if content:
            return content
        reasoning = msg.get("reasoning") or ""
        if reasoning:
            return f"<reasoning>\n{reasoning}\n</reasoning>"
        return ""

    async def stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        http = await self._ensure()
        normalized = [
            (m if isinstance(m, dict) else {"role": m.role, "content": m.content}) for m in messages
        ]
        body = {
            "model": model or self.served_model,
            "messages": normalized,
            "stream": True,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        async with http.stream(
            "POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=body
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[len("data:") :].strip()
                if payload == "[DONE]":
                    return
                try:
                    import orjson

                    parsed = orjson.loads(payload)
                    delta = parsed["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue

    async def health(self) -> bool:
        http = await self._ensure()
        try:
            r = await http.get(f"{self.base_url}/models", headers=self._headers())
            return r.status_code == 200
        except httpx.HTTPError:
            return False
