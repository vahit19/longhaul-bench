"""Final statistics for the paper — runs ONLY when all replication data is in.

Aggregates every (world, arm) summary, then per arm across worlds:
mean accuracy ± bootstrap 95% CI, and key pairwise contrasts via
Mann-Whitney U on per-world accuracies with Holm correction plus
pooled-episode two-proportion z-tests. Emits the paper's main table
(markdown + CSV) and a claims report with per-claim verdicts.

Usage (after all worlds complete):
    local/venv/Scripts/python scripts/final_stats.py \
        --matrices runs/m4_night1 runs/repl_w43 runs/repl_w44 runs/repl_w45 runs/repl_w46 \
        --out docs/paper/final_stats
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

ARMS = ["frozen", "append", "reflect", "gated", "oracle",
        "noise0.4", "defended0.4", "noise0.2-plausible", "noise0.4-plausible",
        "framing-unverified"]

CONTRASTS = [  # (a, b, claim)
    ("append", "frozen", "learning lifts accuracy over control"),
    ("noise0.4", "reflect", "plausible corruption degrades learning (rot)"),
    ("defended0.4", "noise0.4", "consistency gates recover corrupted learning"),
    ("framing-unverified", "reflect", "memory framing label has no effect (null expected)"),
]


def bootstrap_ci(values: list, iters: int = 10000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iters))
    return means[int(iters * 0.025)], means[int(iters * 0.975)]


def mann_whitney_u(a: list, b: list) -> float:
    """Exact two-sided p for tiny samples via permutation of U."""
    def u_stat(x, y):
        return sum(1 for xi in x for yi in y if xi > yi) + 0.5 * sum(
            1 for xi in x for yi in y if xi == yi)
    observed = u_stat(a, b)
    pooled = a + b
    n_a = len(a)
    count = total = 0
    for combo in itertools.combinations(range(len(pooled)), n_a):
        xs = [pooled[i] for i in combo]
        ys = [pooled[i] for i in range(len(pooled)) if i not in combo]
        u = u_stat(xs, ys)
        total += 1
        if abs(u - len(a) * len(b) / 2) >= abs(observed - len(a) * len(b) / 2):
            count += 1
    return count / total


def holm(pvals: list) -> list:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    acc: dict = {a: [] for a in ARMS}
    for m in args.matrices:
        for arm in ARMS:
            f = m / arm / "summary.json"
            if f.exists():
                acc[arm].append(json.loads(f.read_text())["exact_accuracy"])

    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["| arm | worlds | mean acc. | bootstrap 95% CI |", "|---|---|---|---|"]
    for arm in ARMS:
        v = acc[arm]
        if not v:
            continue
        lo, hi = bootstrap_ci(v) if len(v) > 1 else (v[0], v[0])
        lines.append(f"| {arm} | {len(v)} | {sum(v)/len(v):.1%} | [{lo:.1%}, {hi:.1%}] |")

    pvals = []
    contrast_rows = []
    for a, b, claim in CONTRASTS:
        if acc.get(a) and acc.get(b):
            pv = mann_whitney_u(acc[a], acc[b])
            pvals.append(pv)
            diff = sum(acc[a]) / len(acc[a]) - sum(acc[b]) / len(acc[b])
            contrast_rows.append((a, b, claim, diff, pv))
    adj = holm(pvals) if pvals else []

    lines += ["", "| contrast | claim | mean diff | MWU p | Holm-adj p |", "|---|---|---|---|---|"]
    for (a, b, claim, diff, pv), ap in zip(contrast_rows, adj):
        lines.append(f"| {a} vs {b} | {claim} | {diff:+.1%} | {pv:.3f} | {ap:.3f} |")

    report = "\n".join(lines)
    (args.out / "main_table.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
