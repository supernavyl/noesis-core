-- NOESIS Postgres bootstrap
-- Run automatically by docker-compose on first start

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Episodic memory — every interaction
CREATE TABLE IF NOT EXISTS episodic (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content         TEXT NOT NULL,
    embedding       vector(4096),
    tokens          INT,
    model           TEXT,
    reasoning_depth INT,
    meta            JSONB
);
CREATE INDEX IF NOT EXISTS episodic_session_ts ON episodic(session_id, ts);
CREATE INDEX IF NOT EXISTS episodic_ts ON episodic(ts);
CREATE INDEX IF NOT EXISTS episodic_content_trgm ON episodic USING gin (content gin_trgm_ops);
-- HNSW index for vector search — created lazily once we have data

-- Audit log — every reasoning trace, every weight update
CREATE TABLE IF NOT EXISTS audit (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type      TEXT NOT NULL,
    actor           TEXT NOT NULL,
    summary         TEXT,
    payload         JSONB,
    constitution_principle_ref TEXT
);
CREATE INDEX IF NOT EXISTS audit_event_type_ts ON audit(event_type, ts);
CREATE INDEX IF NOT EXISTS audit_actor_ts ON audit(actor, ts);

-- Dream cycle log — every nightly run
CREATE TABLE IF NOT EXISTS dream_runs (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running',
    phases_completed JSONB,
    synthetic_pairs INT,
    pairs_kept      INT,
    adapter_path    TEXT,
    eval_baseline   JSONB,
    eval_after      JSONB,
    merged          BOOLEAN,
    rollback_reason TEXT,
    metrics         JSONB
);

-- Ingestion log — track what we've already ingested
CREATE TABLE IF NOT EXISTS ingested (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    url             TEXT,
    title           TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunks          INT,
    bytes           BIGINT,
    UNIQUE(source, source_id)
);
CREATE INDEX IF NOT EXISTS ingested_source_ts ON ingested(source, ingested_at);

-- Tool registry — synthesized + manually-registered tools
CREATE TABLE IF NOT EXISTS tools (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    version         TEXT NOT NULL,
    description     TEXT NOT NULL,
    signature       JSONB NOT NULL,
    implementation  TEXT NOT NULL,
    synthesized     BOOLEAN NOT NULL DEFAULT FALSE,
    verified        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at    TIMESTAMPTZ,
    use_count       BIGINT NOT NULL DEFAULT 0
);

-- Constitutional violations — for monitoring alignment drift
CREATE TABLE IF NOT EXISTS violations (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    principle_id    TEXT NOT NULL,
    severity        TEXT NOT NULL,
    description     TEXT,
    output_ref      BIGINT REFERENCES episodic(id) ON DELETE SET NULL,
    resolved        BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS violations_principle_ts ON violations(principle_id, ts);
