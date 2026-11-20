"""Replay-only analyses on 30 additional generated plants (no language model involved).

Plants: generator seeds 100-129, standard tier (log and symptom dropout 0.3), 1000 episodes,
995 streamed after 5 probe episodes, probes every 50 episodes, feedback seed 7, budget 100,
reflect write operator.

    python environments/generator.py --seed <s> --episodes 1000 --out runs/replay_worlds/s<s>

(1) Maintenance comparison: recall precision of FIFO, LRU, LFU, relative coverage, ENN, and
    support-aware eviction at miscoding rates 0, 0.2, 0.4, 0.6; plants where each policy
    exceeds FIFO and support-aware eviction.
(2) Thinning law (Proposition 2): predicted precision from the clean support distribution
    against replayed precision at p = 0.1 ... 0.6, for FIFO (native) and support-aware eviction.

Usage: python scripts/replay30.py --out docs/paper/replay30.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from cbm_baselines import CBMStore, POLICIES  # noqa: E402
from memory_replay import world_setup, SEED, thinning  # noqa: E402
from recall_precision import load_jsonl  # noqa: E402
from agents.memory import apply_operator  # noqa: E402
import random  # noqa: E402

SEEDS = range(100, 130)
NOISE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
CMP_POLICIES = ("fifo", "lru", "lfu", "rc", "enn", "support")


def replay(setup, all_eps, policy, noise):
    """Return (pi, support list of correct precedents in knowledge-bearing episodes)."""
    sym = lambda ep: [setup["alarm_symptom"][a["code"]] for a in ep["alarms"] if a["code"] in setup["alarm_symptom"]]
    probes, stream = all_eps[:5], all_eps[5:]
    store, rng = CBMStore(100, policy), random.Random(SEED)
    c = w = 0
    sup = []
    for i, ep in enumerate(stream, 1):
        s = sym(ep)
        cases = store.recall(ep["machine_id"], s)
        gt = (ep["ground_truth"]["component"], ep["ground_truth"]["failure_mode"])
        if cases:
            ok = any((x["component"], x["mode"]) == gt for x in cases)
            c += ok
            w += not ok
            sup.append(sum(x["count"] for x in cases if (x["component"], x["mode"]) == gt))
        apply_operator("reflect", store, ep["machine_id"], s, ep["ground_truth"], rng, noise,
                       alternatives=setup["plausible"].get(ep["machine_id"]))
        if policy.startswith("enn") and i % 50 == 0:
            store.enn_edit()
        if i % 50 == 0 or i == len(stream):
            for p in probes:
                store.recall(p["machine_id"], sym(p))
    return c / (c + w), sup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "replay30.json")
    args = ap.parse_args()
    per = {}
    for s in SEEDS:
        wdir = REPO / "runs" / "replay_worlds" / f"s{s}"
        if not (wdir / "episodes.jsonl").exists():
            subprocess.run([sys.executable, str(REPO / "environments/generator.py"), "--seed", str(s),
                            "--episodes", "1000", "--out", str(wdir)], check=True, capture_output=True)
        all_eps = load_jsonl(wdir / "episodes.jsonl")
        setup = world_setup(wdir)
        row = {}
        for pol in CMP_POLICIES:
            clean_sup = None
            for n in NOISE:
                if n not in (0.0, 0.2, 0.4, 0.6) and pol not in ("fifo", "support"):
                    continue
                pi, sup = replay(setup, all_eps, pol, n)
                row[f"{pol}|{n}"] = pi
                if n == 0.0:
                    clean_sup = sup
                elif pol in ("fifo", "support"):
                    row[f"{pol}|{n}|model"] = thinning(clean_sup, n)
        per[s] = row
        print(s, {k: round(v, 3) for k, v in row.items() if "|model" not in k and k.endswith(("|0.0", "|0.4"))}, flush=True)

    summary = {}
    for pol in CMP_POLICIES:
        for n in (0.0, 0.2, 0.4, 0.6):
            v = [per[s][f"{pol}|{n}"] for s in SEEDS]
            summary[f"{pol}|{n}"] = {
                "pi": statistics.mean(v), "sd": statistics.stdev(v),
                "plants_above_fifo": sum(per[s][f"{pol}|{n}"] > per[s][f"fifo|{n}"] for s in SEEDS),
                "plants_above_support": sum(per[s][f"{pol}|{n}"] > per[s][f"support|{n}"] for s in SEEDS)}
    thin = {}
    for pol in ("fifo", "support"):
        errs = [per[s][f"{pol}|{n}|model"] - per[s][f"{pol}|{n}"] for s in SEEDS for n in NOISE[1:]]
        thin[pol] = {"cells": len(errs), "mae": statistics.mean(abs(e) for e in errs),
                     "max_abs": max(abs(e) for e in errs), "model_above": sum(e > 0 for e in errs)}
    out = {"seeds": list(SEEDS), "per_plant": per, "summary": summary, "thinning": thin}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    for k, v in summary.items():
        print(f"{k:12s} pi={v['pi']:.3f} sd={v['sd']:.3f} >fifo {v['plants_above_fifo']}/30 >support {v['plants_above_support']}/30")
    print("thinning:", thin)


if __name__ == "__main__":
    main()
