"""Combined matrix figure: sliding-window accuracy of all arms on one axis.

Usage:
    python scripts/plot_matrix.py --matrix runs/m4_night1 --window 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito (validated) + gray for the control
SERIES = [
    ("frozen", "#999999", "frozen (control)"),
    ("append", "#0072B2", "append / fifo"),
    ("reflect", "#E69F00", "reflect / compress"),
    ("gated", "#009E73", "gated / importance"),
    ("oracle", "#CC79A7", "oracle (ceiling)"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)

    ends = []
    for name, color, label in SERIES:
        f = args.matrix / name / "results.jsonl"
        if not f.exists():
            continue
        correct = [1 if json.loads(l)["exact_correct"] else 0 for l in f.open(encoding="utf-8")]
        w = min(args.window, len(correct))
        xs = list(range(w, len(correct) + 1))
        ys = [sum(correct[i - w:i]) / w for i in xs]
        ax.plot(xs, ys, color=color, linewidth=1.8)
        ends.append([ys[-1], xs[-1], label, color])

    # stagger direct labels so they never collide
    ends.sort(key=lambda t: t[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 0.045:
            ends[i][0] = ends[i - 1][0] + 0.045
    for y, x, label, color in ends:
        ax.annotate(label, xy=(x, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=color)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel(f"Exact accuracy (window={args.window})", fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.28)  # room for direct labels
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    out = args.matrix / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"matrix_accuracy.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved -> {out}/matrix_accuracy.pdf/.png")


if __name__ == "__main__":
    main()
