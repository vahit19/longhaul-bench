"""Recall-precision decomposition of memory value across all worlds and memory arms.

Generalizes scripts/autopsy.py from one arm on one world to every memory-bearing arm on
all five worlds. No agent is re-run: the analysis reads the released results/traces only.

For each (world, arm) the memory arm's episodes are split by recall quality
(the injected past-case block contains the ground-truth diagnosis or not) and compared
with the frozen control on the SAME episode IDs (paired, deterministic stream):

    pi       = share of memory-bearing episodes whose recall contains the correct precedent
    d_plus   = memory-arm minus frozen accuracy on correct-recall episodes
    d_minus  = memory-arm minus frozen accuracy on misleading-recall episodes
    pi_star  = -d_minus / (d_plus - d_minus)   (break-even recall precision)

and the paired accuracy difference decomposes exactly as
    acc_mem - acc_frozen = w_m * [pi * d_plus + (1 - pi) * d_minus] + w_0 * d_0
with w_m / w_0 the memory-bearing / no-memory shares and d_0 the no-memory difference.

Usage:
    python scripts/recall_precision.py --out docs/paper/recall_precision.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WORLDS = {
    "m4_night1": REPO / "data" / "longhaul-v0-standard" / "episodes.jsonl",
    "repl_w43": REPO / "runs" / "repl_worlds" / "w43" / "episodes.jsonl",
    "repl_w44": REPO / "runs" / "repl_worlds" / "w44" / "episodes.jsonl",
    "repl_w45": REPO / "runs" / "repl_worlds" / "w45" / "episodes.jsonl",
    "repl_w46": REPO / "runs" / "repl_worlds" / "w46" / "episodes.jsonl",
}

# canonical arm -> per-world source dir (same reconciliation as final_stats.py;
# m4_night1 "noise0.4" is the CRUDE negative control and is kept separate)
ARMS = {
    "append": {w: "append" for w in WORLDS},
    "reflect": {w: "reflect" for w in WORLDS},
    "gated": {w: "gated" for w in WORLDS},
    "noise0.2": {"m4_night1": "noise0.2-plausible", **{w: "noise0.2" for w in list(WORLDS)[1:]}},
    "noise0.4": {"m4_night1": "noise0.4-plausible", **{w: "noise0.4" for w in list(WORLDS)[1:]}},
    "defended0.4": {"m4_night1": "noise0.4-defended", **{w: "defended0.4" for w in list(WORLDS)[1:]}},
    "noise0.4-crude": {"m4_night1": "noise0.4"},
}

HEADERS = ("Past confirmed cases", "Past case notes")


def load_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_recall(arm_dir: Path, episodes: dict) -> dict:
    groups = {"correct": [], "misleading": [], "none": []}
    for t in load_jsonl(arm_dir / "traces.jsonl"):
        ep = episodes.get(t["episode_id"])
        if ep is None:
            continue
        user_msg = next((m["content"] for m in t["trace"] if m["role"] == "user"), "")
        idx = max(user_msg.find(h) for h in HEADERS)
        if idx < 0:
            groups["none"].append(t["episode_id"])
            continue
        gt = f"{ep['ground_truth']['component']} ({ep['ground_truth']['failure_mode']})"
        groups["correct" if gt in user_msg[idx:] else "misleading"].append(t["episode_id"])
    return groups


def acc(results: dict, ids: list):
    hits = [results[i]["exact_correct"] for i in ids if i in results]
    return sum(hits) / len(hits) if hits else None


def analyse(world: str, sub: str) -> dict | None:
    arm_dir, frz_dir = REPO / "runs" / world / sub, REPO / "runs" / world / "frozen"
    if not (arm_dir / "traces.jsonl").exists():
        return None
    episodes = {e["episode_id"]: e for e in load_jsonl(WORLDS[world])}
    mem = {r["episode_id"]: r for r in load_jsonl(arm_dir / "results.jsonl")}
    frz = {r["episode_id"]: r for r in load_jsonl(frz_dir / "results.jsonl")}
    g = {k: [i for i in v if i in mem and i in frz] for k, v in split_by_recall(arm_dir, episodes).items()}
    n_mem = len(g["correct"]) + len(g["misleading"])
    n_all = n_mem + len(g["none"])
    out = {"n": n_all, "n_memory": n_mem, "n_correct": len(g["correct"]), "n_misleading": len(g["misleading"])}
    for k in ("correct", "misleading", "none"):
        out[f"acc_mem_{k}"], out[f"acc_frz_{k}"] = acc(mem, g[k]), acc(frz, g[k])
    pi = len(g["correct"]) / n_mem if n_mem else None
    d_plus = out["acc_mem_correct"] - out["acc_frz_correct"] if g["correct"] else None
    d_minus = out["acc_mem_misleading"] - out["acc_frz_misleading"] if g["misleading"] else None
    d_none = out["acc_mem_none"] - out["acc_frz_none"] if g["none"] else 0.0
    all_ids = g["correct"] + g["misleading"] + g["none"]
    observed = acc(mem, all_ids) - acc(frz, all_ids)
    predicted = None
    if pi is not None and d_plus is not None and d_minus is not None:
        predicted = (n_mem / n_all) * (pi * d_plus + (1 - pi) * d_minus) + (len(g["none"]) / n_all) * d_none
    pi_star = -d_minus / (d_plus - d_minus) if d_plus is not None and d_minus is not None and d_plus != d_minus else None
    out.update(pi=pi, d_plus=d_plus, d_minus=d_minus, d_none=d_none, pi_star=pi_star,
               observed_delta=observed, decomposed_delta=predicted)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def transfer_test(per_world: dict) -> dict:
    """Out-of-sample check: predict each corrupted/defended arm's paired delta from the CLEAN
    reflect arm's gain/loss (same world) and the target arm's own recall precision only."""
    cells, pred, obs = [], [], []
    for arm in ("noise0.2", "noise0.4", "defended0.4"):
        for world, x in per_world[arm].items():
            c = per_world["reflect"][world]
            w_m = x["n_memory"] / x["n"]
            p = w_m * (x["pi"] * c["d_plus"] + (1 - x["pi"]) * c["d_minus"]) + (1 - w_m) * c["d_none"]
            cells.append({"arm": arm, "world": world, "predicted": p, "observed": x["observed_delta"]})
            pred.append(p)
            obs.append(x["observed_delta"])
    n = len(pred)
    mp, mo = sum(pred) / n, sum(obs) / n
    cov = sum((a - mp) * (b - mo) for a, b in zip(pred, obs))
    r = cov / (sum((a - mp) ** 2 for a in pred) * sum((b - mo) ** 2 for b in obs)) ** 0.5
    # rot attributable to the precision drop alone, at fixed clean gain/loss
    precision_only = {}
    for arm in ("noise0.2", "noise0.4"):
        att, tot = [], []
        for world, x in per_world[arm].items():
            c = per_world["reflect"][world]
            att.append((c["pi"] - x["pi"]) * (c["d_plus"] - c["d_minus"]))
            tot.append(c["observed_delta"] - x["observed_delta"])
        precision_only[arm] = {"predicted_rot": mean(att), "observed_rot": mean(tot)}
    return {"cells": cells, "n": n, "mae": sum(abs(a - b) for a, b in zip(pred, obs)) / n,
            "pearson_r": r, "sign_agreement": sum((a > 0) == (b > 0) for a, b in zip(pred, obs)),
            "precision_only": precision_only}


LABELS = {
    "append": "append", "reflect": "reflect", "gated": "gated (operator)",
    "noise0.2": "noise$_{0.2}$ (plausible)", "noise0.4": "noise$_{0.4}$ (plausible)",
    "defended0.4": "defended$_{0.4}$", "noise0.4-crude": "crude$_{0.4}$ (1 world)",
}


def latex_rows(report: dict) -> str:
    """Table rows: arm & pi [min, max] & d_plus & d_minus & pi_star & paired delta (points)."""
    lines = []
    for arm, label in LABELS.items():
        m, rows = report["per_arm_mean"][arm], report["per_world"][arm].values()
        pis = [r["pi"] for r in rows]
        rng = f"[{min(pis):.2f}, {max(pis):.2f}]" if len(pis) > 1 else "--"
        lines.append(
            f"{label} & {m['pi']:.3f} & {rng} & ${100 * m['d_plus']:+.1f}$ & ${100 * m['d_minus']:+.1f}$ "
            f"& {m['pi_star']:.2f} & ${100 * m['observed_delta']:+.1f}$ \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "recall_precision.json")
    p.add_argument("--latex", type=Path, default=None, help="optional: write the paper's table rows here")
    args = p.parse_args()
    report = {"per_world": {}, "per_arm_mean": {}}
    for arm, sources in ARMS.items():
        rows = {}
        for world, sub in sources.items():
            r = analyse(world, sub)
            if r is not None:
                rows[world] = r
        report["per_world"][arm] = rows
        report["per_arm_mean"][arm] = {
            k: mean([r[k] for r in rows.values()])
            for k in ("pi", "d_plus", "d_minus", "pi_star", "observed_delta", "decomposed_delta")
        } | {"worlds": len(rows)}
    report["transfer"] = transfer_test(report["per_world"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex_rows(report), encoding="utf-8")
    for arm, m in report["per_arm_mean"].items():
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == "__main__":
    main()
