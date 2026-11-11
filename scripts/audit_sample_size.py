"""Audit sample sizes for monitoring recall precision against the break-even value (Proposition 2).

Decision rule: audit n memory-bearing episodes, flag the memory as harmful when the observed precision
falls below tau = pi_star - gamma/2. Requirement: false-alarm probability <= alpha when pi >= pi_star,
and missed-alarm probability <= alpha when pi <= pi_star - gamma.

Hoeffding bound (distribution-free, closed form): n >= 2 ln(1/alpha) / gamma^2.
Exact binomial: smallest n meeting both error constraints at the boundary values (worst cases),
with the threshold chosen optimally for that n.

Also checks the guarantee empirically by resampling audited recalls from the released traces.

Usage:
    python scripts/audit_sample_size.py --out docs/paper/audit_sample_size.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]


def hoeffding_n(gamma: float, alpha: float) -> int:
    return math.ceil(2 * math.log(1 / alpha) / gamma ** 2)


def exact_n(pi_star: float, gamma: float, alpha: float, n_max: int = 5000):
    p0, p1 = pi_star, pi_star - gamma
    for n in range(1, n_max + 1):
        # flag if X <= k ; false alarm at p0: P(X<=k | p0) ; miss at p1: P(X>k | p1)
        for k in range(n + 1):
            if binom.cdf(k, n, p0) > alpha:
                break
            if binom.sf(k, n, p1) <= alpha:
                return n, k
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "audit_sample_size.json")
    ap.add_argument("--latex", type=Path, default=None, help="optional: write the paper's table rows here")
    args = ap.parse_args()
    rp = json.loads((REPO / "docs" / "paper" / "recall_precision.json").read_text(encoding="utf-8"))
    pi_star = rp["per_arm_mean"]["reflect"]["pi_star"]
    m = rp["per_arm_mean"]["reflect"]
    slope = (m["d_plus"] - m["d_minus"])  # accuracy change per unit precision (omega ~ 1)
    rows = []
    for acc_loss in (0.02, 0.03, 0.05):
        gamma = acc_loss / slope
        for alpha in (0.05, 0.10):
            n_ex, k = exact_n(pi_star, gamma, alpha)
            rows.append({"accuracy_loss_points": 100 * acc_loss, "gamma": gamma, "alpha": alpha,
                         "n_hoeffding": hoeffding_n(gamma, alpha), "n_exact": n_ex, "flag_if_correct_at_most": k})

    # empirical check: resample audits from real per-episode recall labels (clean vs 40% corruption)
    rng = random.Random(0)
    emp = {}
    for row in rows:
        if row["alpha"] != 0.05:
            continue
        n, k = row["n_exact"], row["flag_if_correct_at_most"]
        fa = det = 0
        trials = 2000
        for arm in ("reflect", "noise0.4"):
            for w, x in rp["per_world"][arm].items():
                labels = [1] * x["n_correct"] + [0] * x["n_misleading"]
                flags = sum(sum(rng.choice(labels) for _ in range(n)) <= k for _ in range(trials))
                emp.setdefault(f"{row['accuracy_loss_points']:.0f}pt", {}).setdefault(arm, {})[w] = flags / trials
    report = {"pi_star": pi_star, "accuracy_per_unit_precision": slope, "table": rows, "empirical_flag_rate": emp}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.latex:
        lines = [f"{r['accuracy_loss_points']:.0f} & {r['gamma']:.3f} & {r['alpha']:.2f} & {r['n_hoeffding']} & "
                 f"{r['n_exact']} & {r['flag_if_correct_at_most']} \\\\" for r in rows]
        args.latex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("pi* =", round(pi_star, 3), "slope =", round(slope, 3))
    for r in rows:
        print(r)
    for k, v in emp.items():
        print(k, {a: {w: round(f, 3) for w, f in d.items()} for a, d in v.items()})


if __name__ == "__main__":
    main()
