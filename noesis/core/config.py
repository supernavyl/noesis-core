"""Configuration loader.

Reads YAML configs from ``configs/`` and env vars into typed pydantic models.
Constitutional principles are loaded read-only — they cannot be mutated at runtime.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ServiceSettings(BaseSettings):
    """Environment-driven runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6433"
    qdrant_api_key: str | None = None

    postgres_host: str = "localhost"
    postgres_port: int = 5532
    postgres_db: str = "noesis"
    postgres_user: str = "noesis"
    postgres_password: str = "noesis_dev_only"

    # Default backend: Ollama (already runs on this host with the right models pre-pulled).
    # To swap back to vLLM, set VLLM_BASE_URL=http://localhost:8101/v1
    vllm_base_url: str = "http://localhost:11434/v1"
    vllm_api_key: str = "ollama-no-auth"

    # Ollama provides /v1/embeddings as well (OpenAI-compatible). Reranker is in-process now.
    embed_service_url: str = "http://localhost:11434"
    rerank_service_url: str = "http://localhost:8103"  # TEI reranker (BGE-v2-m3, lives on 3060)

    # Active backend marker — switches model id resolution + endpoint paths.
    backend: str = "ollama"  # "ollama" | "vllm" | "tei"

    noesis_data_dir: Path = PROJECT_ROOT / "data"
    noesis_corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    noesis_checkpoint_dir: Path = PROJECT_ROOT / "data" / "checkpoints"
    noesis_audit_dir: Path = PROJECT_ROOT / "data" / "audit"

    noesis_log_level: str = "INFO"
    noesis_env: str = "development"

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "noesis"
    prometheus_port: int = 9100

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def settings() -> ServiceSettings:
    """Runtime service settings (env-driven)."""
    return ServiceSettings()


@lru_cache(maxsize=1)
def models_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


@lru_cache(maxsize=1)
def ingestion_config() -> dict[str, Any]:
    return _load_yaml("ingestion.yaml")


@lru_cache(maxsize=1)
def dream_config() -> dict[str, Any]:
    return _load_yaml("dream.yaml")


class Constitution(BaseModel):
    """Read-only constitutional principles. Never modified at runtime."""

    model_config = {"arbitrary_types_allowed": True}

    version: str
    last_modified: str
    modifiable_by_model: bool = Field(default=False, frozen=True)
    principles: list[dict[str, Any]]
    prohibitions: list[str]
    refusal: dict[str, Any]

    @classmethod
    def _coerce(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """YAML loads `1.0` as float and dates as datetime.date — coerce to string."""
        coerced = dict(raw)
        if "version" in coerced:
            coerced["version"] = str(coerced["version"])
        if "last_modified" in coerced:
            coerced["last_modified"] = str(coerced["last_modified"])
        return coerced

    def principle(self, principle_id: str) -> dict[str, Any] | None:
        return next((p for p in self.principles if p.get("id") == principle_id), None)


@lru_cache(maxsize=1)
def constitution() -> Constitution:
    raw = _load_yaml("constitution.yaml")
    return Constitution(**Constitution._coerce(raw))


def vault_get(key: str) -> str | None:
    """Pull a secret from claude-vault. Returns None on miss. Never logs the value."""
    import subprocess

    try:
        out = subprocess.run(
            ["claude-vault", "get", key],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout if out.stdout else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def resolve_secret(key: str) -> str | None:
    """Resolve a secret: env first, then vault."""
    if val := os.environ.get(key):
        return val
    return vault_get(f"NOESIS_{key}") or vault_get(key)
