# NOESIS

> νόησις — "the highest form of knowing: direct apprehension of truth"

A self-improving autonomous **knowledge layer** for AI engineering. Hyperquestions everything. Ingests all of AI research continuously. Dreams. Fine-tunes itself. 100% open source. Designed to remain SOTA-relevant on a 5-year horizon.

**License:** MIT

## What NOESIS is — and is not

| NOESIS is | NOESIS is not |
|---|---|
| A reasoning + retrieval layer over a continuously-updated knowledge corpus | A coding assistant or IDE plugin |
| Answers *about* Python/Rust/AI internals, papers, idioms, ecosystems | A code generator that writes your Python or Rust |
| Cross-domain synthesis — connects a Flash Attention paper to a Tokio async primitive to a vLLM prefix-caching detail | An autocomplete tool |
| Surfaces sources, confidence, dissenting research | A black-box chatbot |

The model needs to **know** Python and Rust deeply (idioms, internals, ecosystems), not write them.

## Mission

Build an AI that beats domain PhDs through:
- **Breadth** — all of ML/AI/CS papers + the Python and Rust ecosystems that ship them
- **Currency** — daily ingestion, never stale
- **Synthesis** — cross-domain integration no single human can hold
- **Dream cycle** — overnight self-fine-tuning on filtered synthetic data
- **Hyperquestioning** — Socratic depth no human has patience for

## Status

`v0.1.0` — P0–P5 shipped.

- Hybrid retrieval: BM25 + dense (Qwen3-Embedding-8B) + SPLADE-v3 sparse + BGE cross-encoder rerank
- ReAct + Self-RAG retrieval gates + HyDE query expansion + per-chunk citations
- Ingestion: arxiv, GitHub repos, expert blogs, books, RSS, YouTube, HuggingFace
- Indexes ML/AI papers plus the Python and Rust ecosystems that ship them; point it at any corpus you like via a custom `WorkspaceSource`
- Reasoner: abliterated Qwen3 27B served by Ollama (local, no external API)
- Dream cycle + EWC anchor + constitutional governance (skeleton — see `noesis/dream/`)
- 46 unit tests, CI green on push

Not yet shipped in this release: federation (P6), hosted tier, multi-user auth. See [ROADMAP.md](ROADMAP.md).

## Quickstart

```bash
# Services
docker compose up -d            # qdrant, postgres, grafana

# Python env
uv sync                          # or pip install -e .

# Ingest the corpus
python scripts/ingest_daily.py --bootstrap

# Launch the API
uvicorn noesis.api.server:app --reload --port 8000

# Talk to it
curl http://localhost:8000/v1/chat -d '{"q": "explain Flash Attention 3"}'
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). Ten pillars, model orchestra, 4-tier memory, hierarchical reasoning, sandbox-verified hypotheses, nightly dream cycle, mechanistic self-inspection.

## Add your own source

The four built-in ingestors (arxiv, GitHub, RSS, books, …) cover public AI/ML and the
Python/Rust ecosystems out of the box. To index a corpus of your own — a directory of
markdown/PDF docs, an internal knowledge base, anything on disk — wire a `WorkspaceSource`
into the `IngestionPipeline`:

```bash
python scripts/ingest_example.py /path/to/your/docs --tag my-corpus
```

See [`scripts/ingest_example.py`](scripts/ingest_example.py) for a ~40-line, copy-pasteable
template. Removing or replacing a source never touches the engine — sources are plug-ins.

## Roadmap

See [ROADMAP.md](ROADMAP.md). P0 scaffold → P8 federation. MVP (P0-P3) ships in ~2.5 weeks.

## Hardware

Designed for 4090 (24GB) + 3060 (12GB) = 36GB. Scales up to multi-H100/B100 with config changes. Model slots are config-driven — when a better OSS model drops, change one YAML line.

## License

MIT — see [LICENSE](LICENSE).
