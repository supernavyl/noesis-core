#!/usr/bin/env bash
# NOESIS first-time setup
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Checking prerequisites..."
command -v docker  >/dev/null || { echo "docker not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "    python: $PY_VER (need 3.11+)"

echo "==> Creating data directories..."
mkdir -p data/{corpus/{arxiv,books,forums,github,web},synthetic,checkpoints/{dream_adapters,datasets,logs},episodic,audit,holdout}

echo "==> Bootstrapping Python env..."
if command -v uv >/dev/null; then
    uv sync --extra dev || echo "uv sync failed — fall back to pip"
else
    python3 -m venv .venv
    # shellcheck source=/dev/null
    . .venv/bin/activate
    pip install -U pip
    pip install -e ".[dev]"
fi

echo "==> Pulling Docker images..."
docker compose pull qdrant postgres grafana prometheus otel-collector || true

echo "==> Starting services..."
docker compose up -d qdrant postgres grafana prometheus otel-collector

echo "==> Waiting for Postgres to be ready..."
until docker exec noesis-postgres pg_isready -U noesis -d noesis >/dev/null 2>&1; do
    sleep 1
done

echo "==> Ensuring Qdrant collection..."
python3 -m noesis.knowledge.vector_store || true

cat <<'EOF'

==> NOESIS bootstrap complete.

  Next steps:
    1. Copy docs/env.template.md → .env (edit values)
    2. Run:  make ingest         # bootstrap corpus
    3. Run:  make up             # start API
    4. Visit http://localhost:8000/docs

  Vault keys you'll want to set:
    claude-vault set NOESIS_REDDIT_CLIENT_ID     -
    claude-vault set NOESIS_REDDIT_CLIENT_SECRET -
    claude-vault set NOESIS_HF_TOKEN             -
    claude-vault set NOESIS_GITHUB_TOKEN         -

EOF
