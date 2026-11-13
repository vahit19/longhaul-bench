"""Write-operator x eviction-policy factorial by replaying the memory store (no agent re-run).

The memory update in agents/longrun.py depends only on the episode stream, the feedback
seed, and the operator/policy/defense settings -- never on the agent's answer (the gated
operator consults the memory's own prediction, not the model's). The recalled block of every
episode, and hence whether it contains the correct precedent, can therefore be recomputed
exactly from the released episode files, using the same MemoryStore and apply_operator code.

Step 1 (validation): replay every released memory arm with its original settings and check
that the replayed recall label (correct / misleading / none) of every episode equals the label
read from the released decision traces.

Step 2 (factorial): replay all 3 write operators x 3 eviction policies x 3 miscoding rates on
all five plants and report recall precision pi and memory share omega per cell. With the
conditional effects Delta+/Delta-/Delta0 of the released arm that uses the same write operator
(Proposition 1), each cell's paired accuracy difference against the frozen arm is predicted.

Usage:
    python scripts/memory_replay.py --out docs/paper/memory_replay.json [--latex paper/aei/replay_rows.tex]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from agents.memory import MemoryStore, apply_operator  # noqa: E402
from recall_precision import WORLDS, ARMS, load_jsonl, split_by_recall, analyse  # noqa: E402

SEED = 7
# released arm -> (operator, policy, feedback noise, defense)
ARM_CFG = {
    "append": ("append", "fifo", 0.0, "none"),
    "reflect": ("reflect", "compress", 0.0, "none"),
    "gated": ("gated", "importance", 0.0, "none"),
    "noise0.2": ("reflect", "compress", 0.2, "none"),
    "noise0.4": ("reflect", "compress", 0.4, "none"),
    "defended0.4": ("reflect", "compress", 0.4, "both"),
}
OPERATORS = ("append", "reflect", "gated")
POLICIES = ("fifo", "importance", "compress")
NOISE = (0.0, 0.2, 0.4)
NATIVE = {"append": "fifo", "reflect": "compress", "gated": "importance"}


def world_setup(world_dir: Path) -> dict:
    world = json.loads((world_dir / "world.json").read_text(encoding="utf-8"))
    alarm_symptom = {a["code"]: a["symptom"] for a in world["alarm_table"]}
    manual_rows = {m["machine_id"]: [row for s in m["sections"] for row in s["rows"]]
                   for m in world["manuals"]}
    plausible = {}
    for mid, rows in manual_rows.items():
        pairs = set()
        for row in rows:
            for cause in row["possible_causes"]:
                m2 = re.match(r"(.+) \((.+)\)", cause)
                if m2:
                    pairs.add((m2.group(1), m2.group(2)))
        plausible[mid] = sorted(pairs)
    return {"alarm_symptom": alarm_symptom, "manual_rows": manual_rows, "plausible": plausible}


def replay(setup: dict, all_eps: list, stream_ids: list, probe_size: int, probe_every: int,
           operator: str, policy: str, noise: float, defense: str) -> dict:
    """Return {episode_id: 'correct' | 'misleading' | 'none'} for the stream, as longrun.py would see it."""
    sym = lambda ep: [setup["alarm_symptom"][a["code"]] for a in ep["alarms"]
                      if a["code"] in setup["alarm_symptom"]]

    def consistent(mid, symptoms, comp, mode):
        cause = f"{comp} ({mode})"
        return any(r["symptom"] in symptoms and cause in r["possible_causes"]
                   for r in setup["manual_rows"].get(mid, []))

    by_id = {e["episode_id"]: e for e in all_eps}
    probes = all_eps[:probe_size]
    store = MemoryStore(budget_entries=100, policy=policy)
    rng = random.Random(SEED)
    labels, support = {}, {}

    def recall_block(ep):
        s = sym(ep)
        cases = store.recall(ep["machine_id"], s)
        if defense in ("read", "both"):
            cases = [c for c in cases if consistent(ep["machine_id"], s, c["component"], c["mode"])]
        return cases

    for i, eid in enumerate(stream_ids, 1):
        ep = by_id[eid]
        cases = recall_block(ep)
        gt = (ep["ground_truth"]["component"], ep["ground_truth"]["failure_mode"])
        labels[eid] = ("none" if not cases else
                       "correct" if any((c["component"], c["mode"]) == gt for c in cases) else "misleading")
        support[eid] = sum(c["count"] for c in cases if (c["component"], c["mode"]) == gt)
        s = sym(ep)
        validator = None
        if defense in ("write", "both"):
            validator = lambda c, m, _mid=ep["machine_id"], _s=s: consistent(_mid, _s, c, m)
        if operator != "frozen":
            apply_operator(operator, store, ep["machine_id"], s, ep["ground_truth"], rng, noise,
                           alternatives=setup["plausible"].get(ep["machine_id"]), validator=validator)
        if i % probe_every == 0 or i == len(stream_ids):
            for pe in probes:
                recall_block(pe)  # probes touch 'uses', which the importance policy reads
    return labels, support


def run_params(world: str, all_eps: list, stream_ids: list) -> tuple:
    index = {e["episode_id"]: k for k, e in enumerate(all_eps)}
    probe_size = index[stream_ids[0]]
    # original probe cadence of each world (scripts/chain_*.py); validated in step 1
    probe_every = {"repl_w46": 100}.get(world, 50) if probe_size == 5 else 100
    return probe_size, probe_every


def cell_stats(labels: dict, support: dict) -> dict:
    n = len(labels)
    c = sum(v == "correct" for v in labels.values())
    w = sum(v == "misleading" for v in labels.values())
    s1 = sum(1 for e, v in labels.items() if v != "none" and support[e] == 1)
    sup = [support[e] for e, v in labels.items() if v != "none"]
    return {"pi": c / (c + w) if c + w else None, "omega": (c + w) / n, "q1": s1 / (c + w) if c + w else None,
            "support": sup}


def thinning(sup: list, p: float) -> float:
    """Proposition 2: pi(p) = mean over memory-bearing episodes of 1[S>=1](1 - p^S)."""
    return sum(1 - p ** s for s in sup if s >= 1) / len(sup)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "memory_replay.json")
    p.add_argument("--latex", type=Path, default=None)
    args = p.parse_args()

    report = {"validation": {}, "factorial": {}}
    for world, ep_path in WORLDS.items():
        all_eps = load_jsonl(ep_path)
        setup = world_setup(ep_path.parent)
        episodes = {e["episode_id"]: e for e in all_eps}
        stream_ids = [r["episode_id"] for r in load_jsonl(REPO / "runs" / world / "frozen" / "results.jsonl")]
        probe_size, probe_every = run_params(world, all_eps, stream_ids)

        # step 1: validation against the released traces
        for arm, (op, pol, noise, dfn) in ARM_CFG.items():
            sub = ARMS[arm].get(world)
            arm_dir = REPO / "runs" / world / sub
            if not (arm_dir / "traces.jsonl").exists():
                continue
            groups = split_by_recall(arm_dir, episodes)
            logged = {eid: g for g, ids in groups.items() for eid in ids}
            arm_stream = [r["episode_id"] for r in load_jsonl(arm_dir / "results.jsonl")]
            rep, _ = replay(setup, all_eps, arm_stream, probe_size, probe_every, op, pol, noise, dfn)
            agree = sum(rep[e] == logged.get(e) for e in arm_stream)
            report["validation"].setdefault(world, {})[arm] = {
                "episodes": len(arm_stream), "agree": agree, "probe_size": probe_size,
                "probe_every": probe_every}

        # step 2: factorial
        for op in OPERATORS:
            for pol in POLICIES:
                for noise in NOISE:
                    rep, sup = replay(setup, all_eps, stream_ids, probe_size, probe_every, op, pol, noise, "none")
                    report["factorial"].setdefault(world, {})[f"{op}|{pol}|{noise}"] = cell_stats(rep, sup)

        # conditional effects of the released arm with the same write operator (Proposition 1)
        for op in OPERATORS:
            a = analyse(world, ARMS[op][world])
            report.setdefault("effects", {}).setdefault(world, {})[op] = {
                k: a[k] for k in ("d_plus", "d_minus", "d_none", "pi", "observed_delta")}

    # predicted paired difference per cell: omega*[pi*d+ + (1-pi)*d-] + (1-omega)*d0
    cells = {}
    for world, fac in report["factorial"].items():
        for key, st in fac.items():
            op, pol, _ = key.split("|")
            # append + compress merges identical records and reproduces the reflect knowledge base
            # exactly, so it takes the conditional effects of the released reflect arm
            e = report["effects"][world]["reflect" if (op, pol) == ("append", "compress") else op]
            d0 = e.get("d_none") or 0.0
            pred = st["omega"] * (st["pi"] * e["d_plus"] + (1 - st["pi"]) * e["d_minus"]) + (1 - st["omega"]) * d0
            st["pred_diff"] = pred
            cells.setdefault(key, []).append(st)
    report["mean"] = {k: {"pi": statistics.mean(c["pi"] for c in v),
                          "pi_min": min(c["pi"] for c in v), "pi_max": max(c["pi"] for c in v),
                          "q1": statistics.mean(c["q1"] for c in v),
                          "pred_diff": statistics.mean(c["pred_diff"] for c in v)}
                      for k, v in cells.items()}
    # Proposition 2 on every operator x policy cell: clean support distribution -> pi at p = 0.2, 0.4
    errs = []
    for world, fac in report["factorial"].items():
        for op in OPERATORS:
            for pol in POLICIES:
                sup = fac[f"{op}|{pol}|0.0"]["support"]
                for n in NOISE[1:]:
                    st = fac[f"{op}|{pol}|{n}"]
                    st["pi_model"] = thinning(sup, n)
                    errs.append((op, pol, n, st["pi_model"] - st["pi"]))
    report["thinning_check"] = {
        "cells": len(errs), "mae": statistics.mean(abs(e[3]) for e in errs),
        "max_abs": max(abs(e[3]) for e in errs), "model_above": sum(e[3] > 0 for e in errs),
        "by_operator": {op: statistics.mean(abs(e[3]) for e in errs if e[0] == op) for op in OPERATORS},
        "by_policy": {pol: statistics.mean(abs(e[3]) for e in errs if e[1] == pol) for pol in POLICIES}}
    for fac in report["factorial"].values():
        for st in fac.values():
            st.pop("support", None)
    # per-plant sign check: does each policy raise precision over the operator's native policy?
    report["plants_above_native"] = {
        f"{op}|{pol}|{n}": sum(report["factorial"][w][f"{op}|{pol}|{n}"]["pi"] >
                               report["factorial"][w][f"{op}|{NATIVE[op]}|{n}"]["pi"] for w in report["factorial"])
        for op in OPERATORS for pol in POLICIES for n in NOISE if pol != NATIVE[op]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tot = sum(v["episodes"] for w in report["validation"].values() for v in w.values())
    ok = sum(v["agree"] for w in report["validation"].values() for v in w.values())
    print(f"validation: {ok}/{tot} episode labels reproduced")
    for w, arms in report["validation"].items():
        print(" ", w, {a: f"{v['agree']}/{v['episodes']}" for a, v in arms.items()})
    for k, v in report["mean"].items():
        print(f"{k:28s} pi={v['pi']:.3f} [{v['pi_min']:.3f},{v['pi_max']:.3f}] q1={v['q1']:.3f} pred={100 * v['pred_diff']:+.1f}")
    print("thinning check:", report["thinning_check"])
    print("plants with higher pi than native policy:", report["plants_above_native"])

    if args.latex:
        rows = []
        for op in OPERATORS:
            for pol in POLICIES:
                vals = [report["mean"][f"{op}|{pol}|{n}"] for n in NOISE]
                mark = "$^{\\dagger}$" if NATIVE[op] == pol else ""
                name = {"fifo": "FIFO"}.get(pol, pol)
                rows.append(f"{op} & {name}{mark} & " + " & ".join(f"{v['pi']:.3f}" for v in vals)
                            + f" & {vals[0]['q1']:.2f} & ${100 * vals[0]['pred_diff']:+.1f}$ \\\\")
        args.latex.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("wrote", args.latex)


if __name__ == "__main__":
    main()
