"""Final 5-seed statistics with EXPLICIT arm-source reconciliation.

The development world (m4_night1 = seed 42) named its plausible-poison and
defended arms differently, and additionally has a crude-poison arm that must
NOT be merged with the plausible-poison arms. This script maps each canonical
arm to its correct source directory per world, so no crude/plausible mixing
occurs. Inference is at the world (seed) level per council requirement.

Usage:
    python scripts/final_stats.py --out docs/paper/final_stats
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORLDS = ["m4_night1", "repl_w43", "repl_w44", "repl_w45", "repl_w46"]

# canonical arm -> {world: source subdir}. Missing world = arm absent there.
SOURCES = {
    "frozen":      {w: "frozen" for w in WORLDS},
    "append":      {w: "append" for w in WORLDS},
    "reflect":     {w: "reflect" for w in WORLDS},
    "gated":       {w: "gated" for w in WORLDS},
    "oracle":      {w: "oracle" for w in WORLDS},
    # plausible poison — m4_night1 used the *-plausible dir names
    "noise0.2":    {"m4_night1": "noise0.2-plausible", "repl_w43": "noise0.2",
                    "repl_w44": "noise0.2", "repl_w45": "noise0.2", "repl_w46": "noise0.2"},
    "noise0.4":    {"m4_night1": "noise0.4-plausible", "repl_w43": "noise0.4",
                    "repl_w44": "noise0.4", "repl_w45": "noise0.4", "repl_w46": "noise0.4"},
    "defended0.4": {"m4_night1": "noise0.4-defended", "repl_w43": "defended0.4",
                    "repl_w44": "defended0.4", "repl_w45": "defended0.4", "repl_w46": "defended0.4"},
    # framing ablation not run in m4_night1 -> 4 worlds
    "framing-unverified": {w: "framing-unverified" for w in ["repl_w43", "repl_w44", "repl_w45", "repl_w46"]},
    # crude poison: negative control, single world only — reported separately
    "noise0.4-crude": {"m4_night1": "noise0.4"},
}

CONTRASTS = [
    ("append", "frozen", "learning lifts accuracy over frozen control"),
    ("oracle", "frozen", "manual-oracle ceiling above control"),
    ("noise0.4", "reflect", "plausible corruption degrades learning (rot)"),
    ("noise0.2", "reflect", "rot at lower dose"),
    ("defended0.4", "noise0.4", "consistency gates recover corrupted learning"),
    ("noise0.4-crude", "reflect", "crude poison is inert (negative control)"),
    ("framing-unverified", "reflect", "memory-framing label has no effect (null)"),
]


def arm_accuracies(arm: str) -> list:
    out = []
    for world, sub in SOURCES[arm].items():
        f = REPO / "runs" / world / sub / "summary.json"
        if f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8"))["exact_accuracy"])
    return out


def bootstrap_ci(values, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    m = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iters))
    return m[int(iters * 0.025)], m[int(iters * 0.975)]


def mwu_p(a, b):
    def u(x, y):
        return sum(1 for xi in x for yi in y if xi > yi) + 0.5 * sum(1 for xi in x for yi in y if xi == yi)
    obs = u(a, b)
    pooled, na, cnt, tot = a + b, len(a), 0, 0
    for combo in itertools.combinations(range(len(pooled)), na):
        xs = [pooled[i] for i in combo]
        ys = [pooled[i] for i in range(len(pooled)) if i not in combo]
        tot += 1
        if abs(u(xs, ys) - len(a) * len(b) / 2) >= abs(obs - len(a) * len(b) / 2):
            cnt += 1
    return cnt / tot


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "docs/paper/final_stats")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    acc = {arm: arm_accuracies(arm) for arm in SOURCES}
    lines = ["## Per-arm accuracy (world-level, 5 seeds unless noted)", "",
             "| arm | worlds | mean | bootstrap 95% CI |", "|---|---|---|---|"]
    for arm, v in acc.items():
        if not v:
            continue
        lo, hi = bootstrap_ci(v) if len(v) > 1 else (v[0], v[0])
        lines.append(f"| {arm} | {len(v)} | {sum(v)/len(v):.1%} | [{lo:.1%}, {hi:.1%}] |")

    rows, ps = [], []
    for a, b, claim in CONTRASTS:
        if len(acc.get(a, [])) >= 2 and len(acc.get(b, [])) >= 2:
            pv = mwu_p(acc[a], acc[b])
            ps.append(pv)
            rows.append([a, b, claim,
                         sum(acc[a]) / len(acc[a]) - sum(acc[b]) / len(acc[b]),
                         cliffs_delta(acc[a], acc[b]), pv])
    adj = holm(ps)
    lines += ["", "## Confirmatory contrasts (Mann-Whitney U, Holm-corrected family)", "",
              "| contrast | claim | Δmean | Cliff's δ | MWU p | Holm-adj p | verdict |",
              "|---|---|---|---|---|---|---|"]
    for (a, b, claim, dm, cd, pv), ap in zip(rows, adj):
        verdict = "significant" if ap < 0.05 else ("null (expected)" if "null" in claim or "inert" in claim else "n.s.")
        lines.append(f"| {a} vs {b} | {claim} | {dm:+.1%} | {cd:+.2f} | {pv:.3f} | {ap:.3f} | {verdict} |")

    report = "\n".join(lines)
    (args.out / "main_table.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
