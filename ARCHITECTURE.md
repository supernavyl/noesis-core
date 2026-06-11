# NOESIS Architecture

## Vision

NOESIS is not a chatbot wrapped around a model. It is a **cognitive system** with memory, sleep, hypothesis-testing, self-inspection, and self-modification. The system is designed to remain SOTA-relevant for 5 years by being model-agnostic: every slot is a config entry, and the orchestration layer adapts as frontier OSS models drop.

## The Ten Pillars

### 1. Multi-Model Orchestra
A single monolith cannot beat all domain specialists. NOESIS routes each query to the right combination of:
- **REASONER** — large reasoning model (MoE preferred for VRAM efficiency)
- **CODER** — code-specialized
- **JUDGE** — small fast model for quality scoring
- **CRITIC** — adversarial fine-tune for Socratic challenges
- **SYNTHESIZER** — long-context model for consolidation
- **EMBEDDER** — retrieval embeddings
- **RERANKER** — cross-encoder for retrieval refinement

Routing is dynamic. See `configs/models.yaml` and `noesis/core/orchestrator.py`.

### 2. Hierarchical Memory (4 Stores)
Inspired by cognitive science:
- **Working** — current context (in-token, no persistence)
- **Episodic** — interaction history, events (Postgres + pgvector)
- **Semantic** — facts, concepts, ingested corpus (Qdrant hybrid)
- **Procedural** — skills, tool patterns, compiled prompts (SQLite + FTS5)

Cross-memory consolidation happens during the dream cycle (episodic → semantic).

### 3. Recursive Hyperquestioning (5 Levels)
Reasoning depth scales with query difficulty (test-time compute scaling):
- **L1 ReAct** — think → act → observe (default)
- **L2 Socratic** — challenge every step ("Why? Evidence? Counter?")
- **L3 Tree-of-Thoughts** — parallel branches when uncertain
- **L4 Hypothetical** — generate counterfactuals, sandbox-test them
- **L5 Constitutional** — final self-review against the constitution

### 4. Continuous Knowledge Ingestion
The flywheel that keeps the model fresher than any PhD:
- **arxiv** daily (cs.LG, cs.AI, cs.CL, stat.ML)
- **GitHub** watch (top AI repos for commits + PRs)
- **Reddit** (r/LocalLLaMA, r/MachineLearning, r/artificial)
- **Hacker News** (AI/ML tagged threads)
- **Blogs** (Lilian Weng, Karpathy, Anthropic, Sebastian Raschka, …) via RSS
- **Books** (PDF/EPUB via PyMuPDF + ebooklib)
- **YouTube** (transcripts: Stanford CS lectures, Karpathy Zero-to-Hero)
- **Conferences** (NeurIPS, ICML, ICLR, ACL proceedings)

Pipeline: source → chunk (semantic boundary) → embed (Qwen3-Embedding-8B) → index (Qdrant + SPLADE) → cross-reference.

### 5. Dream Cycle
Nightly, when idle. Inspired by hippocampal replay + sleep consolidation:
- **NREM**: replay recent episodic memories, extract durable facts → semantic store
- **REM**: generate hypothetical scenarios, explore counterfactuals
- **Synthesis**: STaR-style Q&A generation from corpus
- **Filter**: judge model scores synthetic pairs, keep top 10%
- **Train**: Unsloth LoRA fine-tune on filtered data
- **Eval**: holdout benchmark, only merge if metrics ≥ baseline
- **EWC**: Elastic Weight Consolidation to prevent catastrophic forgetting

Guardrails: every merge is reversible. Audit log of every weight update.

### 6. Sandbox Reality-Testing
Hypotheses must be empirically verified, not just plausibly reasoned:
- **Code sandbox** — Docker + Jupyter kernel, runs generated code
- **Paper replication** — implements claims from ingested papers, checks results
- **Benchmark sandbox** — runs model on standard evals (MMLU, GSM8K, HumanEval, etc.)
- **Adversarial sandbox** — generates adversarial inputs, checks robustness

### 7. Mechanistic Self-Inspection
The 5-year-forward pillar most systems lack:
- **Sparse autoencoders** trained on activations → human-interpretable features
- **Activation steering** — surgical behavioral edits without retraining
- **Circuit identification** — which subnetworks do what
- **Refusal direction monitoring** — detect alignment drift after each dream cycle

### 8. Constitutional Self-Governance
Built-in alignment, fully transparent:
- **Constitution** — explicit principles document (`configs/constitution.yaml`)
- **Self-critique** — every output critiqued against constitution
- **Audit log** — every reasoning trace logged
- **Refusal calibration** — refuses only on explicit constitution violations
- **Power preservation** — model cannot modify its own constitution

### 9. Tool Synthesis
Static toolsets get stale. NOESIS writes its own:
- Encounters problem without a suitable tool → designs tool → implements → tests → registers
- Tool library compounds over time
- Every synthesized tool is verified in the sandbox before registration

### 10. Federation-Ready
Multi-machine, multi-node architecture from day one:
- Each node can specialize (one for reasoner, one for ingestion, one for dream)
- Consensus protocols for high-stakes decisions
- Local-first, federation optional

## Hardware Plan (4090 + 3060 = 36GB)

```
RTX 4090 (24GB) — primary compute
├─ Active reasoner: Huihui-Qwen3.6-35B-A3B-abliterated FP8 (~18GB MoE, 3B active)
├─ Hot-swap to coder when needed (Qwen3-Coder-Next-abliterated)
└─ Dream-cycle LoRA training (Unsloth, time-multiplexed)

RTX 3060 (12GB) — auxiliary compute
├─ Embedder: Qwen3-Embedding-8B (~4GB, always loaded)
├─ Reranker: BGE-Reranker-v2-Gemma (~2GB, always loaded)
└─ Critic / fast reasoner: DeepSeek-R1-0528-Qwen3-8B-abliterated (~4GB FP8)
```

5-year upgrade path: when hardware swaps to 2x 5090 / H100 / B100, only `configs/models.yaml` changes — architecture stays.

## Stack Summary

| Layer | Tool |
|---|---|
| Inference serving | vLLM (primary), llama.cpp (fallback) |
| Fine-tuning | Unsloth (LoRA) + TRL (DPO/GRPO) |
| Vector DB | Qdrant (hybrid search native) |
| Episodic store | Postgres + pgvector |
| Procedural store | SQLite + FTS5 |
| Sandbox | Docker SDK + Jupyter kernel manager |
| Ingestion | crawl4ai, arxiv-py, praw, PyMuPDF, ebooklib, youtube-transcript-api, feedparser |
| Orchestration | Custom Python + DSPy (compiled prompts) |
| Interpretability | TransformerLens, SAELens, nnsight |
| Observability | OpenTelemetry + TimescaleDB + Grafana |
| API | FastAPI (OpenAI-compatible) + WebSocket streaming |
| Abliteration | OBLITERATUS toolkit (for custom slot abliteration) |

## Honesty about What's Frontier

| Component | Status |
|---|---|
| Model orchestra, hybrid RAG, ReAct, ToT, ingestion, sandbox, constitutional | Solved engineering |
| STaR self-fine-tuning + judge filter | Research-frontier (quality filter is the hard part) |
| EWC for no-forgetting | Frontier (works for narrow domains) |
| SAE-based steering | Frontier-mature (Anthropic-style reproducible) |
| Tool synthesis with verification | Frontier (verification is the hard part) |
| Multi-node federation | Research (defer to Year 2+) |

~70% solved engineering, ~30% research frontier with understood failure modes.

## What NOESIS Will Not Promise

- "Beats every specialist always" — beats them on breadth + currency + cross-domain synthesis. A narrow PhD on their exact niche of expertise wins head-to-head on novel research questions in their lane.
- "Recursive self-improvement to AGI" — strict eval gates, bounded improvement, no FOOM.
- "Fully autonomous" — constitutional governance + audit log + CEO approval gate on every weight merge.
