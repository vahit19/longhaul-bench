"""Incomplete-manual study on five plants: learning gain and the cost of consistency safeguards.

Plants: the original gap world (seed 47, runs/gap_study) and g43-g46 (runs/gap_study_g43..g46,
generated with the same manual-gap rate of 0.3; see scripts/chain_gap5.py). For each plant the
script reports the share of streamed episodes whose fault is missing from the manual, the
accuracy of frozen, reflect, and reflect with both safeguards, the safeguards' write rejections
and filtered recalls, and plant-level tests (exact Mann-Whitney U, Holm, Cliff's delta).

Usage: python scripts/gap_stats.py --out docs/paper/gap_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from final_stats import mwu_p, holm, cliffs_delta, bootstrap_ci  # noqa: E402

PLANTS = {"g47": ("gap_study", "runs/gap_world"),
          **{f"g{s}": (f"gap_study_g{s}", f"runs/gap_worlds/g{s}") for s in (43, 44, 45, 46)}}
ARMS = ("frozen", "reflect", "reflect-gated")


def load(p: Path) -> list:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "gap_stats.json")
    args = ap.parse_args()
    per = {}
    for plant, (run, wdir) in PLANTS.items():
        if not all((REPO / "runs" / run / a / "summary.json").exists() for a in ARMS):
            continue
        world = json.loads((REPO / wdir / "world.json").read_text(encoding="utf-8"))
        gaps = {(m["machine_id"], c, mo) for m in world["manuals"] for c, mo in m["manual_gaps"]}
        eps = {e["episode_id"]: e for e in load(REPO / wdir / "episodes.jsonl")}
        stream = [r["episode_id"] for r in load(REPO / "runs" / run / "frozen" / "results.jsonl")]
        out = sum((eps[i]["machine_id"], eps[i]["ground_truth"]["component"],
                   eps[i]["ground_truth"]["failure_mode"]) in gaps for i in stream) / len(stream)
        row = {"out_of_manual_share": out, "omitted_pairs": len(gaps)}
        is_out = {i: (eps[i]["machine_id"], eps[i]["ground_truth"]["component"],
                      eps[i]["ground_truth"]["failure_mode"]) in gaps for i in stream}
        for a in ARMS:
            s = json.loads((REPO / "runs" / run / a / "summary.json").read_text(encoding="utf-8"))
            row[a] = s["exact_accuracy"]
            res = load(REPO / "runs" / run / a / "results.jsonl")
            o = [r["exact_correct"] for r in res if is_out.get(r["episode_id"])]
            i_ = [r["exact_correct"] for r in res if r["episode_id"] in is_out and not is_out[r["episode_id"]]]
            row[f"{a}_out"] = sum(o) / len(o) if o else None
            row[f"{a}_in"] = sum(i_) / len(i_) if i_ else None
            if a == "reflect-gated":
                row["gate_stats"] = s.get("gate_stats")
        per[plant] = row

    contrasts = {}
    for name, a, b in (("learning", "reflect", "frozen"), ("safeguards", "reflect-gated", "reflect")):
        x, y = [per[p][a] for p in per], [per[p][b] for p in per]
        d = [xi - yi for xi, yi in zip(x, y)]
        contrasts[name] = {"plants": len(d), "mean_diff": statistics.mean(d), "ci95": bootstrap_ci(d),
                           "plants_same_sign": max(sum(v > 0 for v in d), sum(v < 0 for v in d)),
                           "p": mwu_p(x, y), "cliffs_delta": cliffs_delta(x, y)}
    for n, q in zip(contrasts, holm([c["p"] for c in contrasts.values()])):
        contrasts[n]["p_holm"] = q
    # does the safeguard cost grow with the share of faults missing from the manual?
    xs = [per[p]["out_of_manual_share"] for p in per]
    ys = [per[p]["reflect-gated"] - per[p]["reflect"] for p in per]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (
        sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    report = {"per_plant": per, "contrasts": contrasts, "pearson_gap_share_vs_safeguard_effect": r,
              "mean": {k: statistics.mean(per[p][k] for p in per) for k in
                       [a + sfx for a in ARMS for sfx in ("", "_in", "_out")]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for p, v in per.items():
        print(p, f"gap share {v['out_of_manual_share']:.3f}", {a: round(100 * v[a], 1) for a in ARMS}, v.get("gate_stats"))
    for n, c in contrasts.items():
        print(f"{n:10s} {100 * c['mean_diff']:+.1f} CI [{100 * c['ci95'][0]:+.1f},{100 * c['ci95'][1]:+.1f}] "
              f"same sign {c['plants_same_sign']}/{c['plants']} p={c['p']:.4f} holm={c['p_holm']:.4f} delta={c['cliffs_delta']:+.2f}")
    print("r(gap share, safeguard effect) =", round(r, 3))
    print("mean (all / in-manual / out-of-manual):", {k: round(100 * v, 1) for k, v in report["mean"].items()})


if __name__ == "__main__":
    main()
