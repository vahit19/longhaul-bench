"""Publication-grade figures from a LongHaul-Bench run.

Reads results.jsonl and produces vector PDFs (paper) + PNGs (README):
  1. accuracy_curve   — sliding-window exact accuracy over episodes
  2. latency          — per-episode latency with p50/p95 reference lines
  3. tokens_hist      — tokens-per-episode distribution

Style: colorblind-safe Okabe-Ito subset (validated), single axis, thin marks,
recessive grid, direct labels.

Usage:
    python scripts/plot_run.py --run runs/m2_smoke --window 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#999999"


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#444444", labelsize=9)


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()

    results = [json.loads(l) for l in (args.run / "results.jsonl").open(encoding="utf-8")]
    out = args.run / "figures"
    out.mkdir(exist_ok=True)
    n = len(results)
    print(f"{n} episodes -> {out}")

    # 1 — sliding-window exact accuracy
    correct = [1 if r["exact_correct"] else 0 for r in results]
    w = min(args.window, n)
    xs = list(range(w, n + 1))
    ys = [sum(correct[i - w:i]) / w for i in xs]
    overall = sum(correct) / n

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    style_ax(ax)
    ax.plot(xs, ys, color=BLUE, linewidth=2)
    ax.axhline(overall, color=GRAY, linewidth=1, linestyle="--")
    ax.annotate(f"overall {overall:.0%}", xy=(xs[-1], overall), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=9, color="#444444")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel(f"Exact accuracy (window={w})", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    save(fig, out, "accuracy_curve")

    # 2 — latency per episode with p50/p95
    lat = [r["latency_s"] for r in results]
    lat_sorted = sorted(lat)
    p50 = lat_sorted[n // 2]
    p95 = lat_sorted[max(int(n * 0.95) - 1, 0)]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    style_ax(ax)
    ax.plot(range(1, n + 1), lat, color=BLUE, linewidth=0.8, alpha=0.55)
    for val, label, color in ((p50, f"p50 {p50:.1f}s", GREEN), (p95, f"p95 {p95:.1f}s", ORANGE)):
        ax.axhline(val, color=color, linewidth=1.4)
        ax.annotate(label, xy=(n, val), xytext=(-4, 5), textcoords="offset points",
                    ha="right", fontsize=9, color="#444444")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Latency (s)", fontsize=10)
    save(fig, out, "latency")

    # 3 — tokens per episode
    tokens = [r["tokens"]["prompt"] + r["tokens"]["completion"] for r in results]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    style_ax(ax)
    ax.hist(tokens, bins=20, color=BLUE, edgecolor="white", linewidth=0.8)
    mean_t = sum(tokens) / n
    ax.axvline(mean_t, color=GRAY, linewidth=1, linestyle="--")
    ax.annotate(f"mean {mean_t:.0f}", xy=(mean_t, ax.get_ylim()[1]), xytext=(5, -12),
                textcoords="offset points", fontsize=9, color="#444444")
    ax.set_xlabel("Tokens per episode", fontsize=10)
    ax.set_ylabel("Episodes", fontsize=10)
    save(fig, out, "tokens_hist")


if __name__ == "__main__":
    main()
