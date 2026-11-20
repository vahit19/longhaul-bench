"""Support-aware eviction run with the agent: observed accuracy against the replay prediction.

For each of the five plants, compares the reflect arm with support-aware (importance) eviction,
run with the agent (runs/support_qwen_<w>; Qwen2.5-3B Q4_K_M, same build, GPU host), with the
frozen control and the released reflect arm (compress eviction, CPU host), at 0% and 40%
plausible miscoding. The prediction is Eq. (1) with the replayed recall precision and share of
the support-aware configuration and the conditional effects of the released reflect arm
(docs/paper/cbm_baselines.json). Also reports the recall precision measured on the agent traces.

Usage: python scripts/support_agent.py --out docs/paper/support_agent.json
"""
import argparse, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from recall_precision import load_jsonl, split_by_recall, WORLDS  # noqa: E402
from final_stats import mwu_p, cliffs_delta  # noqa: E402

PLANTS = {"w42": "m4_night1", "w43": "repl_w43", "w44": "repl_w44", "w45": "repl_w45", "w46": "repl_w46"}
NOISE_DIR = {"m4_night1": "noise0.4-plausible"}


def acc(d):
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))["exact_accuracy"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=REPO / "docs/paper/support_agent.json")
    args = ap.parse_args()
    cbm = json.loads((REPO / "docs/paper/cbm_baselines.json").read_text(encoding="utf-8"))["per_plant"]
    rows = {}
    for w, m in PLANTS.items():
        rd, sd = REPO / "runs" / m, REPO / "runs" / f"support_qwen_{w}"
        eps = {e["episode_id"]: e for e in load_jsonl(WORLDS[m])}
        row = {"frozen": acc(rd / "frozen"), "reflect": acc(rd / "reflect"),
               "noise": acc(rd / NOISE_DIR.get(m, "noise0.4")),
               "support": acc(sd / "reflect-support"), "noise_support": acc(sd / "noise0.4-support"),
               "pred_support": cbm[m]["support|0.0"]["pred_diff"], "pred_noise_support": cbm[m]["support|0.4"]["pred_diff"],
               "pi_replay_support": cbm[m]["support|0.0"]["pi"], "pi_replay_noise_support": cbm[m]["support|0.4"]["pi"]}
        for k, sub in (("pi_agent_support", "reflect-support"), ("pi_agent_noise_support", "noise0.4-support")):
            g = split_by_recall(sd / sub, eps)
            row[k] = len(g["correct"]) / (len(g["correct"]) + len(g["misleading"]))
        row["obs_support"] = row["support"] - row["frozen"]
        row["obs_noise_support"] = row["noise_support"] - row["frozen"]
        rows[w] = row
        print(w, {k: round(100 * v, 1) if "pi" not in k else round(v, 3) for k, v in row.items()})
    keys = list(next(iter(rows.values())))
    mean = {k: statistics.mean(r[k] for r in rows.values()) for k in keys}
    tests = {}
    for name, a, b in (("support_vs_reflect", "support", "reflect"), ("noise_support_vs_noise", "noise_support", "noise"),
                       ("support_vs_frozen", "support", "frozen"), ("noise_support_vs_frozen", "noise_support", "frozen")):
        x, y = [r[a] for r in rows.values()], [r[b] for r in rows.values()]
        d = [xi - yi for xi, yi in zip(x, y)]
        tests[name] = {"mean_diff": statistics.mean(d), "min": min(d), "max": max(d),
                       "plants_positive": sum(v > 0 for v in d), "p_mwu": mwu_p(x, y), "cliffs_delta": cliffs_delta(x, y)}
    print("mean", {k: round(100 * v, 1) if "pi" not in k else round(v, 3) for k, v in mean.items()})
    for k, v in tests.items():
        print(k, {kk: round(100 * vv, 1) if kk in ("mean_diff", "min", "max") else round(vv, 4) for kk, vv in v.items()})
    args.out.write_text(json.dumps({"per_plant": rows, "mean": mean, "tests": tests}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
