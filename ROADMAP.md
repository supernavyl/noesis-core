# NOESIS Roadmap

## Phase Plan

| Phase | Scope | Duration | Status |
|---|---|---|---|
| **P0** Scaffold | Project structure, configs, Docker, model registry | 1 day | ⏳ in progress |
| **P1** Knowledge | Ingestion + Qdrant + hybrid retrieval | 1 week | — |
| **P2** Reasoning | ReAct + Socratic + ToT + constitutional | 1 week | — |
| **P3** Sandbox | Docker exec + Jupyter + hypothesis testing | 3 days | — |
| **P4** Memory | 4-tier memory with consolidation | 1 week | — |
| **P5** Dream | STaR + judge filter + Unsloth LoRA + eval gate | 2 weeks | — |
| **P6** Interp | SAE + activation steering + audit | 2 weeks | — |
| **P7** Tool synth | Self-coding tools + verification | 1 week | — |
| **P8** Federation | Multi-node + consensus | 2 weeks | — |

**MVP** = P0 + P1 + P2 + P3 (~2.5 weeks). At MVP, NOESIS already beats specialists on breadth + currency + synthesis.

## P0 — Scaffold (this commit)

- [x] Directory structure
- [x] README.md + ARCHITECTURE.md + ROADMAP.md
- [ ] pyproject.toml with all deps
- [ ] docker-compose.yml (Qdrant + Postgres + Grafana)
- [ ] configs/models.yaml (May 2026 abliterated picks)
- [ ] configs/ingestion.yaml
- [ ] configs/dream.yaml
- [ ] configs/constitution.yaml
- [ ] noesis/ Python module skeleton
- [ ] scripts/setup.sh
- [ ] Makefile
- [ ] .env.example + .gitignore
- [ ] git init + first commit

## P1 — Knowledge (next)

- [ ] Qdrant collection setup (hybrid: dense + sparse)
- [ ] Chunking strategy (semantic, parent-child)
- [ ] Embedder service (Qwen3-Embedding-8B on 3060)
- [ ] Reranker service (BGE-Reranker-v2-Gemma on 3060)
- [ ] Ingestion pipeline base class
- [ ] arxiv ingestor (daily batch, last 30d bootstrap)
- [ ] RSS ingestor (Lilian Weng, Karpathy, Anthropic blog, …)
- [ ] Reddit ingestor (top of week from priority subs)
- [ ] GitHub watch (commits from priority repos)
- [ ] PDF book ingestor (PyMuPDF)
- [ ] YouTube transcript ingestor
- [ ] HuggingFace model card ingestor (catch new releases)
- [ ] Hybrid retrieval: BM25 + dense + RRF + rerank
- [ ] CLI: `noesis ask "..."` — single-shot query over corpus

## P2 — Reasoning

- [ ] vLLM serving (reasoner on 4090)
- [ ] Model orchestrator with hot-swap
- [ ] ReAct loop with tool calling
- [ ] Socratic critique pass (L2)
- [ ] Tree of Thoughts (L3)
- [ ] Constitutional self-review (L5)
- [ ] Test-time compute scaling (depth adaptive)
- [ ] DSPy compiled prompts for each reasoning level

## P3 — Sandbox

- [ ] Docker SDK executor (Python, JS, Rust, Bash)
- [ ] Jupyter kernel manager (persistent state)
- [ ] Hypothesis engine (L4 — generate, simulate, verify)
- [ ] Resource limits + timeout enforcement
- [ ] Output sanitization + result extraction

## P4 — Memory

- [ ] Postgres schema (episodic)
- [ ] SQLite + FTS5 schema (procedural)
- [ ] Memory write hooks on every interaction
- [ ] Cross-memory query
- [ ] Importance scoring (decide what's worth consolidating)

## P5 — Dream Cycle

- [ ] NREM: episodic replay + fact extraction → semantic
- [ ] REM: counterfactual generation
- [ ] STaR Q&A synthesis from semantic store
- [ ] Judge filter (Phi-4 or similar scoring model)
- [ ] Unsloth LoRA training pipeline
- [ ] Holdout eval (MMLU subset + custom AI-engineering bench)
- [ ] EWC implementation (preserve Fisher information matrix)
- [ ] Adapter merging with rollback
- [ ] Cron / systemd timer for nightly cycle

## P6 — Mechanistic Interpretability

- [ ] TransformerLens integration
- [ ] Train SAEs on residual stream activations
- [ ] Feature labeling pipeline (judge auto-labels SAE features)
- [ ] Steering vector library
- [ ] Refusal direction monitor (alert on drift post-dream)

## P7 — Tool Synthesis

- [ ] Tool registry (capability descriptions)
- [ ] Tool design pipeline (problem → spec → implementation → tests)
- [ ] Sandbox verification gate
- [ ] Auto-doc generation for new tools
- [ ] Tool deprecation / version management

## P8 — Federation

- [ ] gRPC inter-node protocol
- [ ] Node discovery + health checks
- [ ] Workload distribution (reasoner-node, ingest-node, dream-node)
- [ ] Consensus protocol for shared state updates
- [ ] Multi-tenant isolation

## Decision Log (live)

- 2026-05-11: Primary reasoner = `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated` (MoE 35B/3B, 18GB FP8)
- 2026-05-11: Embedder = `Qwen/Qwen3-Embedding-8B` (not abliterated — irrelevant for retrieval)
- 2026-05-11: Reranker = `BAAI/bge-reranker-v2-gemma`
- 2026-05-11: Vector DB = Qdrant (hybrid search native)
- 2026-05-11: Episodic store = Postgres + pgvector
- 2026-05-11: Inference = vLLM primary, llama.cpp fallback
- 2026-05-11: Project name = NOESIS (νόησις, highest knowing)
