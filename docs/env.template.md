# Environment Template

Copy this block into `.env` at the project root and fill in values. Real secrets must come from `claude-vault`, not this file.

```bash
# ──── Services (remapped from defaults to avoid host port collisions) ────
QDRANT_URL=http://localhost:6433
QDRANT_API_KEY=

POSTGRES_HOST=localhost
POSTGRES_PORT=5532
POSTGRES_DB=noesis
POSTGRES_USER=noesis
POSTGRES_PASSWORD=noesis_dev_only

# ──── Inference ───────────────────────────────────────────────────────────
VLLM_BASE_URL=http://localhost:8101/v1
VLLM_API_KEY=local-no-auth

EMBED_SERVICE_URL=http://localhost:8102
RERANK_SERVICE_URL=http://localhost:8103

# ──── Ingestion (use claude-vault for real values) ────────────────────────
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=noesis-ingest/0.0.1
HF_TOKEN=

# ──── Paths ───────────────────────────────────────────────────────────────
NOESIS_DATA_DIR=./data
NOESIS_CORPUS_DIR=./data/corpus
NOESIS_CHECKPOINT_DIR=./data/checkpoints
NOESIS_AUDIT_DIR=./data/audit

# ──── Runtime ─────────────────────────────────────────────────────────────
NOESIS_LOG_LEVEL=INFO
NOESIS_ENV=development
CUDA_VISIBLE_DEVICES=0,1

# ──── Observability ───────────────────────────────────────────────────────
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=noesis
PROMETHEUS_PORT=9100
```

## Secret Loading

The runtime loads `.env` first, then overlays values from `claude-vault` for any of these keys:

- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- `HF_TOKEN`
- `QDRANT_API_KEY`
- Any key prefixed `NOESIS_SECRET_*`

To set a secret in the vault:

```bash
claude-vault set NOESIS_REDDIT_CLIENT_ID  -    # paste value via stdin
claude-vault set NOESIS_HF_TOKEN          -
```
