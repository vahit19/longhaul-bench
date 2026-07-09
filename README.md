# LongHaul-Bench: Long-Horizon Reliability of Industrial Edge Agents

> **Research question:** Does an industrial diagnostic agent's knowledge state *improve* or *degrade* over 1000+ task iterations — and which improvement operators keep it reliable under edge-hardware constraints?

**Status:** Design phase (v0). Author: Vahit Feryad, PhD.

## Why this benchmark exists

Agent benchmarks today measure single-episode success on web and coding tasks, on cloud-scale models, with unlimited compute. Real industrial agents live in a different world:

- **Offline** — no cloud API; a quantized local SLM does the reasoning.
- **Long-lived** — the agent runs for months, accumulating experience across thousands of diagnostic episodes.
- **Constrained** — CPU-only or Jetson-class hardware; RAM and latency budgets are hard limits.
- **High-stakes** — a corrupted knowledge base gives confidently wrong maintenance advice.

Nobody measures whether the *self-improvement* mechanisms proposed in recent literature (experience-driven memory, reflection, metric-driven prompt optimization) stay reliable in this regime. LongHaul-Bench does.

## What is measured

Over N ≥ 1000 sequential diagnostic episodes in a synthetic industrial environment:

| Axis | Metric |
|---|---|
| Task performance | success rate curve, time-to-diagnosis |
| Knowledge integrity | corruption rate, contradiction count, stale-fact retention |
| Drift | performance on a frozen probe set re-run every K episodes |
| Edge cost | p50/p95 latency, peak RAM, tokens/episode — on CPU and Jetson |
| Safety | fallback-trigger correctness, confidently-wrong rate |

## Compared improvement operators

1. **Append-only memory** (naive baseline) — every episode's summary is stored.
2. **Reflection → structured experience** (MUSE-style) — trajectories distilled into typed records before integration.
3. **Metric-driven optimization** (DSPy/GEPA-style) — operators updated only when a held-out eval improves.
4. **Frozen agent** (control) — no learning; isolates environment drift from agent drift.

## Data sources & provenance

**No proprietary or employer data is used — by design.** A publishable, reproducible benchmark requires data anyone can regenerate and audit; confidential plant data would make the work both legally risky and scientifically unverifiable.

- **Synthetic core:** machine manuals, alarm-code tables, maintenance history, and HMI/PLC-style logs are procedurally generated from templates with a hidden ground-truth causal model, so diagnostic correctness is objectively scoreable.
- **Realism grounding (public sources only — planned, M1-b):** fault-frequency statistics and degradation patterns will be calibrated against public datasets (UCI AI4I 2020 Predictive Maintenance, NASA C-MAPSS) and public vendor-manual formats. Current v0 templates are hand-designed; this calibration is a declared open item, not yet wired into the generator.
- Every run's environment is fully determined by a random seed → anyone can reproduce the exact 1000-episode world.

## Stack & reproducibility

- **Agent (M2+):** tool-using diagnostic agent — `alarm_lookup`, `manual_search`, `maintenance_history`, `log_fetch` — running a local quantized SLM (llama.cpp, 4-bit GGUF: Qwen2.5-3B, Phi-3.5-mini). Core loop kept dependency-light for edge; a LangGraph implementation is included as a comparison arm (framework overhead is itself measured).
- **Retrieval:** vector search over manual excerpts — local Qdrant (Docker) with a small embedding model; RAG quality tracked with faithfulness/context-precision metrics (DeepEval/Ragas) in the analysis appendix.
- FastAPI tool services
- Eval harness: [Inspect AI](https://inspect.aisi.org.uk/) task definitions and scorers; behavioral auditing with Petri
- **Execution in a resource-capped Linux container** (Docker, e.g. `--memory=8g --cpus=4`): enforces the constrained-device budget identically on any host and makes runs reproducible for the report
- Hardware targets: x86 CPU (8GB RAM budget) → real edge devices via Qualcomm AI Hub (Dragonwing RB3 Gen 2 industrial kit, QCS proxies, Snapdragon X); Jetson = future work

## First results — M2 smoke run (100 episodes, standard tier, CPU)

| Agent | Component acc. | Exact acc. | Latency p50 / p95 | Tokens/ep | Anomaly rate |
|---|---|---|---|---|---|
| Heuristic floor (LLM-free)¹ | — | **86.3%** | <0.01 s | 0 | 0% |
| Qwen2.5-3B Q4_K_M, frozen agent² | 73% | **65%** | 6.3 s / 7.7 s | 1502 | 0% |

¹ 1000 episodes. ² 100 episodes, tool-loop agent (mean 2.05 tool calls/ep), llama.cpp on a 14-core laptop CPU, no learning.

![Sliding-window exact accuracy](runs/m2_smoke/figures/accuracy_curve.png)

**Early observations.** (a) The quantized 3B agent *underperforms* the domain-heuristic floor by 21 points — which sharpens the benchmark's central question: can experience accumulation (improvement operators) close this gap without corrupting the knowledge base? (b) ~6.3 s/episode on laptop CPU confirms edge feasibility for non-interactive diagnostic workloads. (c) The apparent upward trend in the curve is episode-mix variation (the agent is frozen); quantifying such variation is exactly why the full protocol uses 5 seeds and frozen probe sets.

### M3 smoke matrix (10 episodes/config — directional only, full matrix in M4)

| run | exact acc. | memory hit | p50 latency | note |
|---|---|---|---|---|
| frozen (control) | 60% | — | 4.63 s | no learning |
| append / fifo | 70% | 50% | 3.71 s | |
| reflect / compress | 70% | 50% | 3.94 s | 7 rules, 1 KB |
| gated / importance | 70% | 50% | 3.62 s | |
| **oracle** (authoritative row given) | **70%** | 100% | 3.71 s | ceiling ≈ model reasoning, not retrieval |
| reflect + 40% corrupted feedback | 70% | 50% | 4.11 s | corruption invisible at 10 eps — needs horizon |

Directional signals: operators lift accuracy +10 pts over frozen AND cut latency ~20% (past cases → fewer tool calls). Most interesting: the **manual-oracle only reaches 70%** — on this world, retrieval is not the bottleneck; 3B-model reasoning is.

### M4 night matrix — 995 episodes/arm on `longhaul-v0-standard` (single seed)

| Arm | Exact acc. (95% CI) | Memory hit | p50 latency | Evictions (budget 100) |
|---|---|---|---|---|
| frozen (control) | 58.2% ± 3.1 | — | 5.96 s | — |
| append / fifo | 61.1% ± 3.0 | 99% | 4.42 s | 895 |
| reflect / compress | 59.5% ± 3.1 | 99% | 4.39 s | 552 |
| gated / importance | 57.3% ± 3.1 | 99% | 4.28 s | 272 |
| **oracle (ceiling)** | **75.0% ± 2.7** | 100% | 4.11 s | — |
| reflect + 40% corrupted feedback | *(re-running after a transient server timeout at ep. 241; transport hardened)* | | | |

![Matrix accuracy curves](runs/m4_night1/figures/matrix_accuracy.png)

**Headline findings (single seed — 5-seed replication pending):**
1. **Memory is a double-edged sword.** Trace autopsies on the append arm: when recalled cases contain the correct diagnosis (769/987 episodes) accuracy is **70.9%** — near the oracle ceiling; when recall is misleading (218/987) it collapses to **27.5%**, far below the no-memory control. Net operator gains (+1–3 pts, within CI overlap) are the sum of these two large opposing effects; recall precision, not the improvement operator, is the binding lever.
2. **Latency benefit is robust:** all memory arms cut p50 latency ~25% (fewer tool calls when past cases are in context).
3. **The memory budget binds hard:** append evicted 895/995 experiences under FIFO; reflect's compression cut evictions by 38%.
4. **Headroom quantified:** best operator captures ~17% of the frozen→oracle gap (16.8 pts) — current test-time learning leaves most of the available reliability on the table.

Full decision traces (~2.5 MB/arm) are kept out of git; regenerate with `--traces`.

### Model-family robustness (preliminary)

Same protocol, same episodes: **Qwen2.5-3B 65-70% exact / 6.3 s** vs **Phi-3.5-mini 0% exact / 69.8 s, 60% no-diagnosis** — Phi keeps calling tools without committing to a diagnosis. Pending template/prompt-adaptation investigation before drawing conclusions; if it persists, cross-family protocol compliance is itself a benchmark dimension.

## Stack status (honest inventory)

| Component | Status | Where |
|---|---|---|
| Tool-calling agent loop | ✅ implemented | `agents/slm_agent.py` — JSON action protocol, mean 2.05 tool calls/ep measured |
| Quantized SLM runtime (llama.cpp) | ✅ implemented | M2 smoke run, results above |
| Agentic retrieval — keyword & vector modes | ✅ implemented | `manual_search` tool; `LONGHAUL_RETRIEVAL=vector` switches modes (ablation axis) |
| Vector DB (Qdrant, serverless local) + embedding RAG | ✅ implemented | `environments/retrieval.py` — nomic-embed-text via llama.cpp `--embeddings`, no torch/cloud; preliminary 10-ep: vector 50% vs keyword 70% (CIs overlap; full comparison in M4) |
| Inspect AI harness | ✅ implemented | `evals/longhaul_task.py` — dataset/solver/scorer, standard Inspect logs; 10-ep validation: 70% ± 15.3% |
| Improvement operators + memory budget | ✅ implemented | `agents/memory.py` + `agents/longrun.py` — 4 operators × 3 forgetting policies, feedback-noise knob, frozen probes, psutil RSS tracking; 10-ep smoke (reflect/compress): 70% exact, memory hit-rate 50%, probe stable at 80%, p50 4.3s |
| Deterministic retrieval metrics (no judge) | ✅ implemented | `scripts/retrieval_metrics.py` — hit@5 / MRR vs oracle rows; 50-ep: keyword MRR 0.857, vector MRR 0.95 |
| LangGraph comparison arm | ✅ implemented | `agents/langgraph_agent.py` — StateGraph, same protocol; 10-ep: 70% exact, p50 5.96s (no measurable overhead vs bare loop) |
| DeepEval RAG metrics (local judge) | ✅ implemented | `scripts/rag_metrics.py` — contextual relevancy with the local SLM as judge; 5-case vector mode: 0.327 |
| Qualcomm AI Hub integration | ✅ authenticated & verified | `scripts/aihub_profile.py` — live device inventory confirmed, incl. **Dragonwing RB3 Gen 2 Vision Kit** (industrial IoT), QCS8550/8275 proxies, Snapdragon X Elite; first profile jobs → M5 |
| Jetson measurements | future work | no Jetson hardware available; RB3 Gen 2 + Snapdragon via AI Hub cover the real-edge claim |

## Deployment footprint (measured, not estimated)

A key architectural split: the **on-device runtime** (what ships to the edge) vs the **lab-side harness** (evaluation tooling that never leaves the workstation).

| Component | Measured RAM | Where it runs |
|---|---|---|
| llama-server + Qwen2.5-3B Q4 (4k ctx) | 3,650 MB | on the edge device |
| llama-server + nomic-embed (embeddings) | 174 MB | on the edge device |
| Python agent + qdrant-client (in-process) | ~90 MB | on the edge device |
| LangGraph + langchain imports | +38 MB | on the edge device (optional arm) |
| **On-device total** | **≈ 4.0 GB** | fits an 8 GB device with ~4 GB headroom |
| Inspect AI harness (implemented — see Stack status) | +46 MB import | lab workstation, drives the device remotely |
| DeepEval, matplotlib, scipy | — | lab workstation (analysis only) |

Device-class verdict (3B Q4 stack): **8 GB class** (Raspberry Pi 5, Jetson Orin Nano, industrial IPCs) — fits, measured. **4 GB class** — requires the 1.5B model variant (≈1.8 GB total, planned ablation). **< 2 GB** — out of scope (TinyML regime). All runtime components have native ARM64 support (llama.cpp builds, pure-Python qdrant-client/langgraph).

## Roadmap

- [ ] v0.1 — environment generator + frozen-agent baseline, 1000-episode run on CPU
- [ ] v0.2 — operators (1)–(3), comparison report
- [ ] v0.3 — Jetson + Qualcomm AI Hub measurements, quantization ablation (4-bit vs 8-bit)
- [ ] Report — arXiv technical report + blog series

## Architecture

```mermaid
flowchart LR
    G[Seeded world generator] -->|world.json| T[Tool services<br/>alarm_lookup / manual_search / maintenance_history]
    G -->|episodes.jsonl| A[Diagnostic agent<br/>quantized SLM, llama.cpp]
    A <-->|one JSON action per turn| T
    A -->|experience| M[Memory / knowledge state]
    M -->|improvement operators<br/>append / reflect / metric-gated| A
    A -->|diagnosis| S[Scorer vs hidden ground truth]
    S --> R[Reliability metrics<br/>accuracy curve · corruption · drift<br/>latency · RAM · anomaly rate]
```

## Quickstart

Generate a reproducible demo world — pure Python 3.10+, no dependencies:

```bash
python environments/generator.py --machines 5 --episodes 50 --seed 42 --out runs/demo
```

Outputs `world.json` (machines, alarm-code table, manual excerpts for RAG, maintenance history) and `episodes.jsonl` (diagnostic episodes with hidden ground-truth root cause). Same seed → byte-identical world. A pre-generated sample lives in `runs/demo/`.

**Difficulty tiers** (calibrated against the LLM-free heuristic baseline, 1000 episodes, seed 42):

| Tier | `--log-dropout` | `--symptom-dropout` | Baseline exact accuracy |
|---|---|---|---|
| easy | 0.0 | 0.0 | 99.9% |
| standard | 0.3 | 0.3 | 86.3% |
| hard | 0.7 | 0.5 | 73.3% |

Further hardening (multi-fault episodes, overlapping log vocabulary) is tracked in `docs/PAPER_PLAN.md` M1.

## Repository layout

```
environments/   synthetic industrial world generator
agents/         baseline agent, memory module
operators/      improvement operator implementations
evals/          Inspect AI tasks, probe sets, scorers
runs/           experiment configs + result artifacts
docs/           design notes, blog drafts
```
