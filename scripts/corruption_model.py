"""Independent-thinning model of recall precision under plausible feedback corruption.

Model. In the clean (p = 0) reflect arm, let s_e be the total support of the correct diagnosis in the
recalled block of episode e (sum of the "seen Nx" counts of recalled rows whose label is the ground truth;
s_e = 0 for a misleading recall). If each confirmation is independently miscoded with probability p, a
correct row backed by s_e confirmations keeps at least one correct confirmation with probability 1 - p^s_e.
Ignoring displacement of the correct row by corrupted rows (a second-order effect), the predicted precision is

    pi_hat(p) = (1/|E_m|) * sum_{e in E_m} (1 - p^{s_e}),

computed from the clean arm only and compared with the precision observed in the corrupted arms.
The model also predicts the break-even corruption rate p* where pi_hat(p*) equals the clean break-even
precision pi*. No agent is re-run; inputs are the released traces and docs/paper/recall_precision.json.

Usage:
    python scripts/corruption_model.py --out docs/paper/corruption_model.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORLDS = {
    "m4_night1": REPO / "data" / "longhaul-v0-standard" / "episodes.jsonl",
    "repl_w43": REPO / "runs" / "repl_worlds" / "w43" / "episodes.jsonl",
    "repl_w44": REPO / "runs" / "repl_worlds" / "w44" / "episodes.jsonl",
    "repl_w45": REPO / "runs" / "repl_worlds" / "w45" / "episodes.jsonl",
    "repl_w46": REPO / "runs" / "repl_worlds" / "w46" / "episodes.jsonl",
}
HEADERS = ("Past confirmed cases", "Past case notes")
ROW = re.compile(r"->\s*(.+?)\s*\((.+?)\)(?:,\s*seen\s*(\d+)x)?\s*$")


def load_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def supports(world: str, arm: str = "reflect") -> list:
    """Support s_e of the ground truth in each memory-bearing episode's recalled block (clean arm)."""
    episodes = {e["episode_id"]: e for e in load_jsonl(WORLDS[world])}
    out = []
    for t in load_jsonl(REPO / "runs" / world / arm / "traces.jsonl"):
        ep = episodes.get(t["episode_id"])
        if ep is None:
            continue
        user = next((m["content"] for m in t["trace"] if m["role"] == "user"), "")
        idx = max(user.find(h) for h in HEADERS)
        if idx < 0:
            continue
        gt = (ep["ground_truth"]["component"], ep["ground_truth"]["failure_mode"])
        s = 0
        for line in user[idx:].splitlines()[1:]:
            if not line.startswith("- "):
                break
            m = ROW.search(line)
            if m and (m.group(1).strip(), m.group(2).strip()) == gt:
                s += int(m.group(3) or 1)
        out.append(s)
    return out


def pi_hat(s_list: list, p: float) -> float:
    return sum((1 - p ** s) if s > 0 else 0.0 for s in s_list) / len(s_list)


def solve_break_even(s_list: list, target: float) -> float | None:
    lo, hi = 0.0, 1.0
    if pi_hat(s_list, lo) < target:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pi_hat(s_list, mid) > target else (lo, mid)
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "corruption_model.json")
    ap.add_argument("--latex", type=Path, default=None, help="optional: write the paper's table rows here")
    args = ap.parse_args()
    rp = json.loads((REPO / "docs" / "paper" / "recall_precision.json").read_text(encoding="utf-8"))["per_world"]
    report = {"per_world": {}}
    for w in WORLDS:
        s = supports(w)
        pstar = rp["reflect"][w]["pi_star"]
        row = {
            "n_memory": len(s),
            "support_hist": {str(k): sum(1 for x in s if x == k) for k in sorted(set(min(x, 10) for x in s))},
            "pi_clean_observed": rp["reflect"][w]["pi"],
            "pi_clean_model": pi_hat(s, 0.0),
            "pi_0.2_model": pi_hat(s, 0.2), "pi_0.2_observed": rp["noise0.2"][w]["pi"],
            "pi_0.4_model": pi_hat(s, 0.4), "pi_0.4_observed": rp["noise0.4"][w]["pi"],
            "pi_star_clean": pstar,
            "p_break_even_model": solve_break_even(s, pstar),
        }
        # fully a-priori accuracy prediction: clean-arm gain/loss + modelled precision only
        c = rp["reflect"][w]
        omega = c["n_memory"] / c["n"]
        for p in ("0.2", "0.4"):
            pi_m = row[f"pi_{p}_model"]
            row[f"delta_{p}_model"] = omega * (pi_m * c["d_plus"] + (1 - pi_m) * c["d_minus"]) + (1 - omega) * c["d_none"]
            row[f"delta_{p}_observed"] = rp[f"noise{p}"][w]["observed_delta"]
        report["per_world"][w] = row
    keys = ["pi_clean_observed", "pi_clean_model", "pi_0.2_model", "pi_0.2_observed", "pi_0.4_model",
            "pi_0.4_observed", "pi_star_clean", "p_break_even_model"]
    report["mean"] = {k: sum(r[k] for r in report["per_world"].values()) / len(WORLDS) for k in keys}
    errs = [abs(r[f"pi_{p}_model"] - r[f"pi_{p}_observed"]) for r in report["per_world"].values() for p in ("0.2", "0.4")]
    report["mae_precision"] = sum(errs) / len(errs)
    cells = [(r[f"delta_{p}_model"], r[f"delta_{p}_observed"]) for r in report["per_world"].values() for p in ("0.2", "0.4")]
    report["accuracy_a_priori"] = {
        "n": len(cells),
        "mae_points": 100 * sum(abs(a - b) for a, b in cells) / len(cells),
        "sign_agreement": sum((a > 0) == (b > 0) for a, b in cells),
        "mean_model_points": {p: 100 * sum(r[f"delta_{p}_model"] for r in report["per_world"].values()) / len(WORLDS) for p in ("0.2", "0.4")},
        "mean_observed_points": {p: 100 * sum(r[f"delta_{p}_observed"] for r in report["per_world"].values()) / len(WORLDS) for p in ("0.2", "0.4")},
    }
    print("a-priori accuracy:", report["accuracy_a_priori"])
    for w, r in report["per_world"].items():
        r["p_single_support"] = int(r["support_hist"].get("1", 0)) / r["n_memory"]
        r["p_safe"] = 1 - r["pi_star_clean"] / r["pi_clean_observed"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.latex:
        names = {"m4_night1": "P1", "repl_w43": "P2", "repl_w44": "P3", "repl_w45": "P4", "repl_w46": "P5"}
        lines = []
        for w, r in report["per_world"].items():
            lines.append(
                f"{names[w]} & {r['pi_clean_observed']:.3f} & {r['p_single_support']:.2f} & "
                f"{r['pi_0.2_model']:.3f} / {r['pi_0.2_observed']:.3f} & {r['pi_0.4_model']:.3f} / {r['pi_0.4_observed']:.3f} & "
                f"{r['pi_star_clean']:.3f} & {r['p_safe']:.2f} & {r['p_break_even_model']:.2f} \\\\")
        args.latex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for w, r in report["per_world"].items():
        print(w, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("mean", {k: round(v, 3) for k, v in report["mean"].items()}, "MAE", round(report["mae_precision"], 4))


if __name__ == "__main__":
    main()
