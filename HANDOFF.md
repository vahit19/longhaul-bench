# HANDOFF — Project State & Resume Guide

> Purpose: resume the research/code from any machine with zero context loss.
> Scope: technical/reproducibility only. Owner: Vahit Feryad.

## What this project is

**LongHaul-Bench** — a benchmark measuring whether self-improving LLM agents *improve or rot*
over ~1000 diagnostic episodes under industrial-edge constraints (quantized 3B model, hard
experience-memory budget, offline). A paper based on this work is under submission as a preprint.

## Headline results (all reproducible from seeds)

| Finding | Evidence | Status |
|---|---|---|
| Learning lifts accuracy (+3.8% over frozen) | 5 worlds, Cliff's δ=+1.00, Holm p=.048 | significant |
| Plausible corruption → rot (−6.0%) | 5 worlds, dose-dependent (20% n.s.) | significant |
| Consistency gates recover damage (+6.0%) | 5 worlds; NOT "beats clean" (w43/44 reversed) | significant |
| Mechanism: memory ±paired effect | +9.4 correct-recall / −19.8 misleading, episode-matched vs frozen (`scripts/autopsy.py`) | confound-controlled |
| Memory value ∝ 1/capability | Llama-3.2 reflect +28 vs Qwen +2.3; defense neutral/harmful on weak model | 2 worlds |
| Incomplete manual → memory shines | gap world: frozen 47.9% vs reflect 56.9% (+9); gates cut to +3.3 | gap_study |
| Crude poison inert (negative control) | 60.0% ≈ clean | 1 world |
| Oracle ceiling 76.8% vs floor heuristic 86.3% | 3B reasoning is the bottleneck; v0 world easier than intended | honest limitation |

## Where everything lives

- `data/longhaul-v0-standard|hard` — canonical datasets (seed 42); `longhaul-v1-standard` = AI4I-calibrated
- `runs/m4_night1, repl_w43..w46` — 5-world replication (9 arms each); `xfam_*` — cross-family; `gap_study` — incomplete-manual; `runs/ci_capped` — capped-container reproduction
- `scripts/final_stats.py` — THE stats (world-level MWU+Holm, arm-name reconciliation inside); `autopsy.py` — matched mechanism analysis; `footprint.py` — per-process RSS
- `docs/PAPER_PLAN.md`, `docs/RELATED_WORK.md`, `docs/COUNCIL_DECISION.md` — planning/positioning/review notes
- `.github/workflows/` — `capped-run.yml` (8GB/4CPU full arm in public CI), `qai-profile.yml`, `paper-build.yml`

## Council blocking scoreboard

**ALL 8 REVIEW BLOCKERS CLEARED.** Highlights: episode-matched autopsy script (central figure
made causal); full 5-seed stats with Holm correction; incomplete-manual arm quantifies gate
false-block cost; second model family (Llama-3.2) via a strict-vocabulary prompt fix;
capped-run reproduces reflect at 61.2% vs 61.3% uncapped in public CI with a SHA-256-pinned
model. Device profiling (RB3/QCS via Qualcomm AI Hub) parked as future work after documented
toolchain failures — contribution already rescoped, no on-device execution claimed.

## Standing rules (do not violate)

1. NO "Co-Authored-By: Claude" in commits — sole author Vahit Feryad <vahit.feryat@gmail.com>.
2. No paper prose until all results frozen; claims only at measured strength; every number regenerable from seeds; "proven" only post-replication.
3. m4_night1 `noise0.4` = CRUDE poison (negative control) — never merge with repl worlds' plausible `noise0.4` (final_stats.py handles the mapping).
4. Runs launch as DETACHED OS processes (PowerShell Start-Process) — session-layer background tasks get killed.
5. Qwen results use the canonical prompt; cross-family uses strict-vocab (env LONGHAUL_STRICT_VOCAB=1); never mix in one comparison.

## Resume recipe (new machine)

1. Clone repo; `py -3.10 -m venv local/venv && local/venv/Scripts/pip install inspect-ai openai qdrant-client psutil langgraph langchain-openai`
2. Models (gitignored): Qwen2.5-3B-Instruct Q4_K_M + nomic-embed-text-v1.5 Q8 GGUF + llama.cpp b9918 into `local/`
3. Servers: llama-server :8080 (Qwen, `-c 4096`), :8081 (nomic `--embeddings`), :8082 (Llama-3.2 for cross-family)
4. Run state: each experiment dir has `results.jsonl` (progress) + `summary.json` (done marker); the matrix orchestrator resumes by skipping arms that have a summary.json
5. Reproduce stats: `local/venv/Scripts/python scripts/final_stats.py`
