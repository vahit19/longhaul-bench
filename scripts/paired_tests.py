"""Episode-level paired tests for the confirmatory contrasts (complements world-level final_stats.py).

Because the episode stream is deterministic, two arms on the same world answer the SAME episodes,
so each contrast is a matched-pairs comparison. For every world we count discordant pairs
(b: arm A right and arm B wrong, c: A wrong and B right) and run an exact McNemar test; across
worlds we pool the discordant counts (stratified McNemar for matched pairs). No agent is re-run.

Usage:
    python scripts/paired_tests.py --out docs/paper/paired_tests.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
WORLDS = ["m4_night1", "repl_w43", "repl_w44", "repl_w45", "repl_w46"]

# same arm-source reconciliation as final_stats.py
SOURCES = {
    "frozen": {w: "frozen" for w in WORLDS},
    "append": {w: "append" for w in WORLDS},
    "reflect": {w: "reflect" for w in WORLDS},
    "gated": {w: "gated" for w in WORLDS},
    "oracle": {w: "oracle" for w in WORLDS},
    "noise0.2": {"m4_night1": "noise0.2-plausible", **{w: "noise0.2" for w in WORLDS[1:]}},
    "noise0.4": {"m4_night1": "noise0.4-plausible", **{w: "noise0.4" for w in WORLDS[1:]}},
    "defended0.4": {"m4_night1": "noise0.4-defended", **{w: "defended0.4" for w in WORLDS[1:]}},
}

CONTRASTS = [
    ("append", "frozen"),
    ("reflect", "frozen"),
    ("gated", "frozen"),
    ("oracle", "frozen"),
    ("noise0.4", "reflect"),
    ("noise0.2", "reflect"),
    ("defended0.4", "noise0.4"),
]


def load(world: str, arm: str) -> dict:
    f = REPO / "runs" / world / SOURCES[arm][world] / "results.jsonl"
    with f.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["episode_id"]: bool(r["exact_correct"]) for r in rows}


def mcnemar(b: int, c: int) -> float:
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "paired_tests.json")
    args = p.parse_args()
    report = {}
    for a, b_arm in CONTRASTS:
        per_world, B, C, N = {}, 0, 0, 0
        for w in WORLDS:
            ra, rb = load(w, a), load(w, b_arm)
            ids = sorted(set(ra) & set(rb))
            b = sum(ra[i] and not rb[i] for i in ids)
            c = sum(rb[i] and not ra[i] for i in ids)
            per_world[w] = {"n": len(ids), "b": b, "c": c, "delta": (b - c) / len(ids), "p": mcnemar(b, c)}
            B, C, N = B + b, C + c, N + len(ids)
        report[f"{a} vs {b_arm}"] = {
            "per_world": per_world,
            "pooled": {"n": N, "b": B, "c": C, "delta": (B - C) / N, "p": mcnemar(B, C)},
            "worlds_same_sign": sum((v["b"] - v["c"]) * (B - C) > 0 for v in per_world.values()),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in report.items():
        pw = " ".join(f"{x['p']:.3g}" for x in v["per_world"].values())
        print(f"{k:24} pooled n={v['pooled']['n']} b={v['pooled']['b']} c={v['pooled']['c']} "
              f"delta={100 * v['pooled']['delta']:+.2f} p={v['pooled']['p']:.2e} same_sign={v['worlds_same_sign']}/5 | per-world p: {pw}")


if __name__ == "__main__":
    main()
