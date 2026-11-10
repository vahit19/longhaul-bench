"""Figure: recall precision under corruption (per world) and out-of-sample prediction check.

Reads docs/paper/recall_precision.json (written by scripts/recall_precision.py); no agent is re-run.

Usage:
    python scripts/make_precision_figure.py --out paper/isa/figures/Figure_2.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARMS = [("reflect", "clean"), ("noise0.2", "20%"), ("noise0.4", "40%"), ("defended0.4", "40% +\nsafeguards")]
MARKERS = {"noise0.2": ("o", "#0072B2"), "noise0.4": ("s", "#D55E00"), "defended0.4": ("^", "#009E73")}
LABELS = {"noise0.2": "20% corruption", "noise0.4": "40% corruption", "defended0.4": "40% + safeguards"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "paper" / "isa" / "figures" / "Figure_2.pdf")
    args = p.parse_args()
    rep = json.loads((REPO / "docs" / "paper" / "recall_precision.json").read_text(encoding="utf-8"))
    pw, tr = rep["per_world"], rep["transfer"]

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    worlds = list(pw["reflect"])
    xs = range(len(ARMS))
    for w in worlds:
        ax1.plot(xs, [pw[a][w]["pi"] for a, _ in ARMS], color="0.6", lw=0.8, marker="o", ms=3)
    ax1.plot(xs, [rep["per_arm_mean"][a]["pi"] for a, _ in ARMS], color="black", lw=2, marker="o", ms=5,
             label="mean of 5 plants")
    pstar = rep["per_arm_mean"]["reflect"]["pi_star"]
    ax1.axhline(pstar, color="#D55E00", ls="--", lw=1)
    ax1.text(-0.05, pstar - 0.006, f"clean break-even $\\pi^*$ = {pstar:.2f}", ha="left", va="top",
             color="#D55E00", fontsize=8)
    ax1.set_xticks(list(xs), [lab for _, lab in ARMS])
    ax1.set_ylabel("Recall precision $\\pi$")
    ax1.set_title("(a) Recall precision by feedback condition", fontsize=9)
    ax1.legend(loc="lower left", frameon=False, fontsize=8)

    for arm, (m, col) in MARKERS.items():
        cells = [c for c in tr["cells"] if c["arm"] == arm]
        ax2.scatter([100 * c["predicted"] for c in cells], [100 * c["observed"] for c in cells],
                    marker=m, color=col, s=28, label=LABELS[arm])
    lim = [-6, 6]
    ax2.plot(lim, lim, color="0.4", lw=0.8, ls=":")
    ax2.axhline(0, color="0.85", lw=0.6)
    ax2.axvline(0, color="0.85", lw=0.6)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Predicted paired $\\Delta$ (points)")
    ax2.set_ylabel("Observed paired $\\Delta$ (points)")
    ax2.set_title(f"(b) Out-of-sample check (MAE {100 * tr['mae']:.1f} pts, r = {tr['pearson_r']:.2f})", fontsize=9)
    ax2.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
