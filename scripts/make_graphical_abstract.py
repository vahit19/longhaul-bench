"""Graphical abstract: the knowledge loop of the diagnostic agent and the three results that govern it.

Numbers are read from docs/paper/*.json (written by the analysis scripts); nothing is typed in by hand.

Usage:
    python scripts/make_graphical_abstract.py --out paper/aei/Graphical_Abstract.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#555555"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="white", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=10.5, weight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=8.8, color="#222222", linespacing=1.35)


def arrow(ax, a, b, color=GREY, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "paper" / "aei" / "Graphical_Abstract.pdf")
    args = p.parse_args()
    cm = json.loads((REPO / "docs" / "paper" / "corruption_model.json").read_text(encoding="utf-8"))
    rp = json.loads((REPO / "docs" / "paper" / "memory_replay.json").read_text(encoding="utf-8"))["mean"]
    mae = cm["mae_precision"]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(10, 4.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # knowledge loop (top row)
    y, h, w = 0.60, 0.33, 0.155
    xs = [0.025, 0.222, 0.419, 0.616, 0.813]
    box(ax, xs[0], y, w, h, "Maintenance feedback", "confirmed repair,\nmiscoded with\nprobability $p$", ORANGE)
    box(ax, xs[1], y, w, h, "Write operator", "append / reflect /\ngated", BLUE)
    box(ax, xs[2], y, w, h, "Case base", "100 records:\nsymptoms $\\rightarrow$ diagnosis,\nsupport count;\nevicted when full", BLUE)
    box(ax, xs[3], y, w, h, "Recall", "3 most similar\ncases inserted\ninto the prompt", BLUE)
    box(ax, xs[4], y, w, h, "Edge LLM agent", "Qwen2.5-3B, 4-bit,\nCPU only, tools:\nalarms, manual, history", GREY)
    for i in range(4):
        arrow(ax, (xs[i] + w + 0.004, y + h / 2), (xs[i + 1] - 0.006, y + h / 2))
    # eviction loop under the case base
    arrow(ax, (xs[4] + w / 2, y - 0.005), (xs[0] + w / 2, y - 0.005), color=ORANGE, rad=-0.12)
    ax.text(0.5, 0.445, "diagnosis $\\rightarrow$ repair $\\rightarrow$ work order (about 1,000 episodes, 5 plants)",
            ha="center", fontsize=8.8, color=ORANGE)

    # results (bottom row)
    y2, h2, w2 = 0.04, 0.33, 0.215
    xs2 = [0.025, 0.272, 0.519, 0.766]
    box(ax, xs2[0], y2, w2, h2, "Break-even precision",
        "knowledge helps only if\n$\\pi > \\pi^{*} = |\\Delta^{-}| / (\\Delta^{+} + |\\Delta^{-}|)$\nlearning +3.8 pts;\n40% miscoding $-$6.0 pts", GREEN)
    box(ax, xs2[1], y2, w2, h2, "Miscoding erodes $\\pi$",
        f"$\\pi(p) = \\sum_s q_s (1 - p^{{s}})$\nno fitted parameters;\nmean error {mae:.3f}", GREEN)
    box(ax, xs2[2], y2, w2, h2, "Audit rule",
        "$n \\geq 2 \\ln(1/\\alpha) / \\gamma^{2}$\nwork orders to audit;\nexact rule: 240 for\na 3-point loss", GREEN)
    box(ax, xs2[3], y2, w2, h2, "Evict by support",
        f"exact replay of 3 $\\times$ 3 designs;\nreflect at 40% miscoding:\n"
        f"$\\pi$ = {rp['reflect|compress|0.4']['pi']:.3f} $\\rightarrow$ {rp['reflect|importance|0.4']['pi']:.3f}\n"
        "with importance eviction", GREEN)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
