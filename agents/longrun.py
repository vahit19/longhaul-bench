"""Long-horizon experiment driver for LongHaul-Bench (the core experiment).

Runs the SLM agent SEQUENTIALLY through an episode stream. After each episode
the chosen improvement operator updates the knowledge state from outcome
feedback. Every --probe-every episodes, a frozen probe set is re-run WITHOUT
memory writes to measure knowledge-quality drift. Per-episode telemetry:
correctness, latency, tokens, memory entries/bytes/evictions, and RSS of the
agent process plus all llama-server processes (psutil).

Usage (llama-server on :8080; run with local/venv python):
    local/venv/Scripts/python agents/longrun.py \
        --world runs/v01/world.json --episodes runs/v01/episodes.jsonl \
        --operator reflect --policy compress --budget 100 \
        --limit 10 --probe-every 5 --out runs/m3_smoke_reflect
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.memory import MemoryStore, apply_operator  # noqa: E402
from agents.slm_agent import run_episode  # noqa: E402


def system_rss_mb() -> float:
    """Agent process + every llama-server process, MB."""
    total = psutil.Process().memory_info().rss
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if p.info["name"] and "llama-server" in p.info["name"]:
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return round(total / 1e6)


def symptoms_of(ep: dict, alarm_symptom: dict) -> list:
    return [alarm_symptom[a["code"]] for a in ep["alarms"] if a["code"] in alarm_symptom]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--operator", choices=["frozen", "append", "reflect", "gated"], required=True)
    p.add_argument("--policy", choices=["fifo", "importance", "compress"], default="fifo")
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--feedback-noise", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--probe-every", type=int, default=50)
    p.add_argument("--probe-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--oracle", action="store_true",
                   help="upper bound: hand the agent the authoritative manual row")
    p.add_argument("--defense", choices=["none", "read", "write", "both"], default="none",
                   help="rot-mitigation baselines: manual-consistency gates on memory reads/writes")
    p.add_argument("--memory-label", choices=["confirmed", "unverified"], default="confirmed",
                   help="framing ablation: how memory context is presented to the model")
    p.add_argument("--traces", action="store_true",
                   help="dump full decision traces to traces.jsonl (knowledge-state autopsies)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    world = json.loads(args.world.read_text(encoding="utf-8"))
    alarm_symptom = {a["code"]: a["symptom"] for a in world["alarm_table"]}
    all_eps = [json.loads(l) for l in args.episodes.open(encoding="utf-8")]
    probes, stream = all_eps[: args.probe_size], all_eps[args.probe_size: args.probe_size + args.limit]

    store = MemoryStore(budget_entries=args.budget, policy=args.policy)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    manual_rows = {
        m["machine_id"]: [row for s in m["sections"] for row in s["rows"]]
        for m in world["manuals"]
    }
    # valid (component, mode) pairs per machine — plausible-poison alternatives
    import re as _re
    plausible = {}
    for mid, rows in manual_rows.items():
        pairs = set()
        for row in rows:
            for cause in row["possible_causes"]:
                m2 = _re.match(r"(.+) \((.+)\)", cause)
                if m2:
                    pairs.add((m2.group(1), m2.group(2)))
        plausible[mid] = sorted(pairs)

    def oracle_context(ep: dict) -> str:
        gt = f"{ep['ground_truth']['component']} ({ep['ground_truth']['failure_mode']})"
        for row in manual_rows.get(ep["machine_id"], []):
            if gt in row["possible_causes"]:
                return "Authoritative manual row for this fault:\n" + json.dumps(row)
        return ""

    def consistent_with_manual(machine_id: str, symptoms: list, component: str, mode: str) -> bool:
        """Mitigation gate: is (component, mode) a manual-listed cause for ANY observed symptom?"""
        cause = f"{component} ({mode})"
        for row in manual_rows.get(machine_id, []):
            if row["symptom"] in symptoms and cause in row["possible_causes"]:
                return True
        return False

    def run_one(ep: dict, use_memory: bool) -> dict:
        symptoms = symptoms_of(ep, alarm_symptom)
        ctx = ""
        if args.oracle:
            ctx = oracle_context(ep)
        elif use_memory:
            cases = store.recall(ep["machine_id"], symptoms)
            if args.defense in ("read", "both"):  # abstain-on-conflict
                kept = [c for c in cases
                        if consistent_with_manual(ep["machine_id"], symptoms, c["component"], c["mode"])]
                gate_stats["read_filtered"] += len(cases) - len(kept)
                cases = kept
            if cases:
                ctx = store.render(cases, label=args.memory_label)
        r = run_episode(args.endpoint, world, ep, memory_context=ctx, include_trace=args.traces)
        r["memory_used"] = bool(ctx)
        return r

    gate_stats = {"write_rejections": 0, "read_filtered": 0}
    results, probe_curve = [], []
    traces_f = (args.out / "traces.jsonl").open("w", encoding="utf-8") if args.traces else None
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as f:
        for i, ep in enumerate(stream, 1):
            r = run_one(ep, use_memory=True)
            if traces_f is not None:
                traces_f.write(json.dumps({"episode_id": r["episode_id"],
                                           "trace": r.pop("trace", [])}) + "\n")
                traces_f.flush()
            symptoms = symptoms_of(ep, alarm_symptom)
            validator = None
            if args.defense in ("write", "both"):
                def validator(c, m, _mid=ep["machine_id"], _s=symptoms):
                    ok = consistent_with_manual(_mid, _s, c, m)
                    if not ok:
                        gate_stats["write_rejections"] += 1
                    return ok
            apply_operator(args.operator, store, ep["machine_id"], symptoms,
                           ep["ground_truth"], rng, args.feedback_noise,
                           alternatives=plausible.get(ep["machine_id"]), validator=validator)
            r.update({"episode_index": i, **store.stats(), "system_rss_mb": system_rss_mb()})
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()

            if i % args.probe_every == 0 or i == len(stream):
                probe_acc = statistics.mean(
                    run_one(pe, use_memory=True)["exact_correct"] for pe in probes)
                probe_curve.append({"after_episode": i, "probe_exact": probe_acc})
                print(f"[{i}/{len(stream)}] probe={probe_acc:.0%} "
                      f"mem={store.stats()} rss={system_rss_mb()}MB", flush=True)

    summary = {
        "operator": args.operator, "policy": args.policy, "budget": args.budget,
        "feedback_noise": args.feedback_noise, "episodes": len(results),
        "exact_accuracy": statistics.mean(r["exact_correct"] for r in results),
        "memory_hit_rate": statistics.mean(r["memory_used"] for r in results),
        "final_memory": store.stats(),
        "gate_stats": gate_stats,
        "peak_system_rss_mb": max(r["system_rss_mb"] for r in results),
        "latency_p50_s": statistics.median(r["latency_s"] for r in results),
        "probe_curve": probe_curve,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
