"""Replication of the main effects and of the recall-precision theory across model families.

Families (3B parameters, 4-bit quantized):
  qwen     Qwen2.5-3B-Instruct Q4_K_M, canonical prompt (main study; runs m4_night1, repl_w43..w46)
  llama    Llama-3.2-3B-Instruct Q4_K_M, strict-vocabulary prompt (runs xfam_llama_w42..w46)
  qwen_strict  Qwen2.5-3B Q4_K_M with the strict-vocabulary prompt (frozen and reflect; runs xfam_qwen_w42..w46)

Per family and plant it reads the released results and traces (no agent re-run) and reports
accuracy of the frozen, reflect, 40% plausible miscoding, and safeguarded arms; plant-level
contrasts with the exact Mann-Whitney U test, Holm-corrected within family, and Cliff's delta;
and the recall-precision parameters of the reflect and miscoded arms (pi, Delta+, Delta-,
break-even pi*), with a check that the decomposition reproduces the observed paired effect.

Usage: python scripts/cross_family.py --out docs/paper/cross_family.json [--latex rows.tex]
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
from recall_precision import load_jsonl, split_by_recall  # noqa: E402

EPISODES = {
    "w42": REPO / "data" / "longhaul-v0-standard" / "episodes.jsonl",
    **{w: REPO / "runs" / "repl_worlds" / w / "episodes.jsonl" for w in ("w43", "w44", "w45", "w46")},
}
QWEN_DIR = {"w42": "m4_night1", "w43": "repl_w43", "w44": "repl_w44", "w45": "repl_w45", "w46": "repl_w46"}
QWEN_ARM = {"frozen": {}, "reflect": {}, "noise0.4": {"w42": "noise0.4-plausible"},
            "defended0.4": {"w42": "noise0.4-defended"}}
FAMILIES = {
    "qwen": {"label": "Qwen2.5-3B", "dir": lambda w: QWEN_DIR[w], "arm": lambda a, w: QWEN_ARM[a].get(w, a)},
    "llama": {"label": "Llama-3.2-3B", "dir": lambda w: f"xfam_llama_{w}", "arm": lambda a, w: a},
    "qwen_strict": {"label": "Qwen2.5-3B (strict prompt)", "dir": lambda w: f"xfam_qwen_{w}", "arm": lambda a, w: a},
}
ARMS = ("frozen", "reflect", "noise0.4", "defended0.4")
CONTRASTS = (("learning", "reflect", "frozen"), ("miscoding", "noise0.4", "reflect"),
             ("safeguards", "defended0.4", "noise0.4"))


def arm_path(fam: str, arm: str, world: str) -> Path:
    f = FAMILIES[fam]
    return REPO / "runs" / f["dir"](world) / f["arm"](arm, world)


def results(fam, arm, world):
    p = arm_path(fam, arm, world) / "results.jsonl"
    if not (arm_path(fam, arm, world) / "summary.json").exists():
        return None
    return {r["episode_id"]: r for r in load_jsonl(p)}


def decompose(fam, arm, world, episodes):
    mem, frz = results(fam, arm, world), results(fam, "frozen", world)
    if mem is None or frz is None:
        return None
    g = {k: [i for i in v if i in mem and i in frz]
         for k, v in split_by_recall(arm_path(fam, arm, world), episodes).items()}
    acc = lambda res, ids: sum(res[i]["exact_correct"] for i in ids) / len(ids) if ids else None
    n_mem = len(g["correct"]) + len(g["misleading"])
    n_all = n_mem + len(g["none"])
    pi = len(g["correct"]) / n_mem
    dp = acc(mem, g["correct"]) - acc(frz, g["correct"])
    dm = acc(mem, g["misleading"]) - acc(frz, g["misleading"])
    d0 = acc(mem, g["none"]) - acc(frz, g["none"]) if g["none"] else 0.0
    ids = g["correct"] + g["misleading"] + g["none"]
    observed = acc(mem, ids) - acc(frz, ids)
    decomposed = (n_mem / n_all) * (pi * dp + (1 - pi) * dm) + (len(g["none"]) / n_all) * d0
    return {"pi": pi, "d_plus": dp, "d_minus": dm, "d_none": d0, "pi_star": -dm / (dp - dm), "omega": n_mem / n_all,
            "observed": observed, "decomposed": decomposed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "cross_family.json")
    p.add_argument("--latex", type=Path, default=None)
    args = p.parse_args()

    episodes = {w: {e["episode_id"]: e for e in load_jsonl(p)} for w, p in EPISODES.items()}
    report = {}
    for fam in FAMILIES:
        acc = {a: {} for a in ARMS}
        for a in ARMS:
            for w in EPISODES:
                res = results(fam, a, w)
                if res:
                    acc[a][w] = sum(r["exact_correct"] for r in res.values()) / len(res)
        fam_rep = {"accuracy": acc, "plants": {a: len(v) for a, v in acc.items()}, "contrasts": {}, "theory": {}}
        ps = []
        for name, a, b in CONTRASTS:
            ws = [w for w in EPISODES if w in acc[a] and w in acc[b]]
            if len(ws) < 2:
                continue
            x, y = [acc[a][w] for w in ws], [acc[b][w] for w in ws]
            diffs = [xi - yi for xi, yi in zip(x, y)]
            fam_rep["contrasts"][name] = {
                "plants": len(ws), "mean_diff": statistics.mean(diffs), "ci95": bootstrap_ci(diffs),
                "plants_same_sign": max(sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)),
                "p": mwu_p(x, y), "cliffs_delta": cliffs_delta(x, y)}
            ps.append(name)
        adj = holm([fam_rep["contrasts"][n]["p"] for n in ps])
        for n, q in zip(ps, adj):
            fam_rep["contrasts"][n]["p_holm"] = q
        for arm in ("reflect", "noise0.4"):
            cells = {w: decompose(fam, arm, w, episodes[w]) for w in EPISODES}
            cells = {w: c for w, c in cells.items() if c}
            if cells:
                keys = ("pi", "d_plus", "d_minus", "pi_star", "omega", "observed", "decomposed")
                fam_rep["theory"][arm] = {"plants": len(cells), "per_plant": cells,
                                          **{k: statistics.mean(c[k] for c in cells.values()) for k in keys},
                                          "max_abs_decomposition_error": max(abs(c["observed"] - c["decomposed"])
                                                                             for c in cells.values())}
        # format failures: share of episodes with a malformed answer or no diagnosis
        fails = {}
        for a in ARMS:
            n = bad = 0
            for w in EPISODES:
                res = results(fam, a, w)
                if res:
                    n += len(res)
                    bad += sum(r["status"] != "ok" or r.get("anomalies", 0) > 0 for r in res.values())
            if n:
                fails[a] = bad / n
        fam_rep["format_failure_rate"] = fails
        # transfer: predict the 40% miscoding arm from the clean reflect arm's conditional effects
        # and the miscoded arm's own recall precision and share (same plant)
        tr = []
        cl, mc = fam_rep["theory"].get("reflect", {}).get("per_plant", {}), fam_rep["theory"].get("noise0.4", {}).get("per_plant", {})
        for w in set(cl) & set(mc):
            c, x = cl[w], mc[w]
            pred = x["omega"] * (x["pi"] * c["d_plus"] + (1 - x["pi"]) * c["d_minus"]) + (1 - x["omega"]) * c["d_none"]
            tr.append({"plant": w, "predicted": pred, "observed": x["observed"]})
        if tr:
            fam_rep["transfer"] = {"cells": tr, "mae": statistics.mean(abs(t["predicted"] - t["observed"]) for t in tr),
                                   "sign_agree": sum((t["predicted"] > 0) == (t["observed"] > 0) for t in tr)}
        # does recall precision predict the sign of the effect? (pi above or below pi*)
        cells = [c for arm in ("reflect", "noise0.4") for c in fam_rep["theory"].get(arm, {}).get("per_plant", {}).values()]
        fam_rep["sign_rule"] = {"cells": len(cells),
                                "agree": sum((c["pi"] > c["pi_star"]) == (c["observed"] > 0) for c in cells)}
        report[fam] = fam_rep

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for fam, r in report.items():
        print(f"\n== {FAMILIES[fam]['label']}  plants per arm: {r['plants']}")
        for a in ARMS:
            if r["accuracy"][a]:
                print(f"  {a:12s} {100 * statistics.mean(r['accuracy'][a].values()):.1f}%")
        for n, c in r["contrasts"].items():
            print(f"  {n:11s} {100 * c['mean_diff']:+.1f} pts  CI [{100 * c['ci95'][0]:+.1f},{100 * c['ci95'][1]:+.1f}]"
                  f"  same sign {c['plants_same_sign']}/{c['plants']}  p={c['p']:.4f} holm={c['p_holm']:.4f}"
                  f"  delta={c['cliffs_delta']:+.2f}")
        for arm, t in r["theory"].items():
            print(f"  theory[{arm}] plants={t['plants']} pi={t['pi']:.3f} d+={100 * t['d_plus']:+.1f}"
                  f" d-={100 * t['d_minus']:+.1f} pi*={t['pi_star']:.3f} obs={100 * t['observed']:+.1f}"
                  f" max|obs-decomp|={t['max_abs_decomposition_error']:.2e}")
        print(f"  sign rule (pi > pi* <=> gain): {r['sign_rule']['agree']}/{r['sign_rule']['cells']}")
        print("  format failures:", {a: round(v, 4) for a, v in r["format_failure_rate"].items()})
        if "transfer" in r:
            print(f"  transfer clean->40%: MAE {100 * r['transfer']['mae']:.1f} pts, sign {r['transfer']['sign_agree']}/{len(r['transfer']['cells'])}")

    if args.latex:
        rows = []
        for fam, r in report.items():
            if "safeguards" not in r["contrasts"]:
                continue
            m = lambda a: 100 * statistics.mean(r["accuracy"][a].values()) if r["accuracy"][a] else float("nan")
            c = r["contrasts"]
            t = r["theory"].get("reflect", {})
            rows.append(
                f"{FAMILIES[fam]['label']} & {min(r['plants'].values())} & {m('frozen'):.1f} & {m('reflect'):.1f} & "
                f"${100 * c['learning']['mean_diff']:+.1f}$ & ${100 * c['miscoding']['mean_diff']:+.1f}$ & "
                f"${100 * c['safeguards']['mean_diff']:+.1f}$ & {t.get('pi', float('nan')):.3f} & "
                f"{t.get('pi_star', float('nan')):.3f} \\\\")
        args.latex.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("wrote", args.latex)


if __name__ == "__main__":
    main()
