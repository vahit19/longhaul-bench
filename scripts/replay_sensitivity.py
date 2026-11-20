"""Sensitivity of the maintenance comparison to the knowledge budget and to class imbalance.

Exact replay only (no agent re-run), with the reflect write operator, as in cbm_baselines.py.

(1) Budget: B in {25, 50, 100, 200, 400} on the five released plants, at 0% and 40% miscoding.
(2) Imbalance: five AI4I-calibrated plants (generator --calibrated, seeds 42-46; seed 42 equals
    data/longhaul-v1-standard), budget 100, 995 streamed episodes after 5 probe episodes, probes
    every 50 episodes. Recall precision is also reported separately for rare faults (the
    failure modes in the least frequent quarter of each plant) and the other faults.

    python environments/generator.py --seed <s> --episodes 1000 --calibrated --out runs/calibrated_worlds/c<s>

Usage: python scripts/replay_sensitivity.py --out docs/paper/replay_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from cbm_baselines import replay  # noqa: E402
from memory_replay import world_setup, run_params  # noqa: E402
from recall_precision import WORLDS, load_jsonl  # noqa: E402

POLICIES = ("fifo", "lfu", "rc", "support")
BUDGETS = (25, 50, 100, 200, 400)
NOISE = (0.0, 0.4)


def precision(labels: dict, ids) -> float | None:
    c = sum(labels[i] == "correct" for i in ids)
    w = sum(labels[i] == "misleading" for i in ids)
    return c / (c + w) if c + w else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "replay_sensitivity.json")
    args = ap.parse_args()
    rep = {"budget": {}, "imbalance": {}}

    for world, ep_path in WORLDS.items():
        all_eps = load_jsonl(ep_path)
        setup = world_setup(ep_path.parent)
        stream = [r["episode_id"] for r in load_jsonl(REPO / "runs" / world / "frozen" / "results.jsonl")]
        ps, pe = run_params(world, all_eps, stream)
        for b in BUDGETS:
            for pol in POLICIES:
                for n in NOISE:
                    rep["budget"].setdefault(f"{b}|{pol}|{n}", []).append(
                        replay(setup, all_eps, stream, ps, pe, pol, n, budget=b)["pi"])
        print("budget sweep done:", world, flush=True)

    for s in (42, 43, 44, 45, 46):
        wdir = REPO / "runs" / "calibrated_worlds" / f"c{s}"
        all_eps = load_jsonl(wdir / "episodes.jsonl")
        setup = world_setup(wdir)
        stream = [e["episode_id"] for e in all_eps[5:]]
        freq = Counter((e["ground_truth"]["component"], e["ground_truth"]["failure_mode"]) for e in all_eps[5:])
        cut = sorted(freq.values())[len(freq) // 4]
        rare = {k for k, v in freq.items() if v <= cut}
        by_id = {e["episode_id"]: e for e in all_eps}
        is_rare = {i: (by_id[i]["ground_truth"]["component"], by_id[i]["ground_truth"]["failure_mode"]) in rare
                   for i in stream}
        for pol in POLICIES:
            for n in NOISE:
                r = replay(setup, all_eps, stream, 5, 50, pol, n, budget=100, return_labels=True)
                lab = r["labels"]
                rep["imbalance"].setdefault(f"{pol}|{n}", []).append({
                    "pi": r["pi"], "pi_rare": precision(lab, [i for i in stream if is_rare[i]]),
                    "pi_common": precision(lab, [i for i in stream if not is_rare[i]]),
                    "rare_share": sum(is_rare.values()) / len(stream)})
        print("imbalance done: c%d (rare share %.3f)" % (s, sum(is_rare.values()) / len(stream)), flush=True)

    summary = {"budget": {}, "imbalance": {}}
    for k, v in rep["budget"].items():
        b, pol, n = k.split("|")
        ref = rep["budget"][f"{b}|fifo|{n}"]
        summary["budget"][k] = {"pi": statistics.mean(v), "plants_above_fifo": sum(x > y for x, y in zip(v, ref))}
    for k, v in rep["imbalance"].items():
        pol, n = k.split("|")
        ref = rep["imbalance"][f"fifo|{n}"]
        summary["imbalance"][k] = {
            **{m: statistics.mean(c[m] for c in v if c[m] is not None) for m in ("pi", "pi_rare", "pi_common", "rare_share")},
            "plants_above_fifo": sum(c["pi"] > r["pi"] for c, r in zip(v, ref)),
            "plants_above_fifo_rare": sum((c["pi_rare"] or 0) > (r["pi_rare"] or 0) for c, r in zip(v, ref))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"per_plant": rep, "summary": summary}, indent=2), encoding="utf-8")
    print("\nBUDGET (mean pi over 5 plants; plants above FIFO)")
    for b in BUDGETS:
        print(f" B={b:3d} " + "  ".join(
            f"{pol}:{summary['budget'][f'{b}|{pol}|{n}']['pi']:.3f}({summary['budget'][f'{b}|{pol}|{n}']['plants_above_fifo']})"
            for n in NOISE for pol in POLICIES))
    print("\nIMBALANCE (calibrated plants)")
    for k, v in summary["imbalance"].items():
        print(f" {k:12s} pi={v['pi']:.3f} rare={v['pi_rare']:.3f} common={v['pi_common']:.3f} "
              f">fifo {v['plants_above_fifo']}/5 rare>fifo {v['plants_above_fifo_rare']}/5 rare share {v['rare_share']:.3f}")


if __name__ == "__main__":
    main()
