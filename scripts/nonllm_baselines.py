"""Non-LLM reference points on the five released plants (no agent re-run).

  manual-rule   reads only what the agent can read: for the observed symptoms it returns the
                cause listed under the most symptom rows of the machine's manual table (ties:
                first listed). Unlike the 91-line heuristic, it never sees the generator's
                failure-mode table or log signatures.
  cbr-only      the case base alone: the solution of the top recalled case (reflect write
                operator, compress or support-aware eviction, same stream and feedback seed as
                the released arms); with no recalled case it falls back to manual-rule.
  cbr+manual    the top recalled case that the manual lists for an observed symptom, else
                manual-rule (the read-side consistency check without a language model).

Reports exact accuracy per plant and the mean, at 0% and 40% plausible miscoding.

Usage: python scripts/nonllm_baselines.py --out docs/paper/nonllm_baselines.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from agents.memory import MemoryStore, apply_operator  # noqa: E402
from memory_replay import world_setup, run_params, SEED  # noqa: E402
from recall_precision import WORLDS, load_jsonl  # noqa: E402


def parse(cause: str):
    m = re.match(r"(.+) \((.+)\)", cause)
    return (m.group(1), m.group(2)) if m else None


def manual_rule(rows: list, symptoms: list):
    votes, order = Counter(), {}
    for r in rows:
        if r["symptom"] in symptoms:
            for c in r["possible_causes"]:
                pc = parse(c)
                if pc:
                    votes[pc] += 1
                    order.setdefault(pc, len(order))
    if not votes:
        return None
    best = max(votes.values())
    return min((pc for pc, v in votes.items() if v == best), key=lambda pc: order[pc])


def run(world, ep_path, policy, noise):
    all_eps = load_jsonl(ep_path)
    setup = world_setup(ep_path.parent)
    stream = [r["episode_id"] for r in load_jsonl(REPO / "runs" / world / "frozen" / "results.jsonl")]
    ps, pe = run_params(world, all_eps, stream)
    by_id = {e["episode_id"]: e for e in all_eps}
    probes = all_eps[:ps]
    sym = lambda ep: [setup["alarm_symptom"][a["code"]] for a in ep["alarms"] if a["code"] in setup["alarm_symptom"]]
    store, rng = MemoryStore(100, policy), random.Random(SEED)
    hits = Counter()
    for i, eid in enumerate(stream, 1):
        ep = by_id[eid]
        s, mid = sym(ep), ep["machine_id"]
        gt = (ep["ground_truth"]["component"], ep["ground_truth"]["failure_mode"])
        rows = setup["manual_rows"].get(mid, [])
        man = manual_rule(rows, s)
        cases = store.recall(mid, s)
        cbr = (cases[0]["component"], cases[0]["mode"]) if cases else man
        listed = [c for c in cases if any(r["symptom"] in s and f"{c['component']} ({c['mode']})" in r["possible_causes"] for r in rows)]
        cbr_m = (listed[0]["component"], listed[0]["mode"]) if listed else man
        hits["manual-rule"] += man == gt
        hits["cbr-only"] += cbr == gt
        hits["cbr+manual"] += cbr_m == gt
        apply_operator("reflect", store, mid, s, ep["ground_truth"], rng, noise, alternatives=setup["plausible"].get(mid))
        if i % pe == 0 or i == len(stream):
            for p in probes:
                store.recall(p["machine_id"], sym(p))
    return {k: v / len(stream) for k, v in hits.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "nonllm_baselines.json")
    args = ap.parse_args()
    res = {}
    for policy in ("compress", "importance"):
        for noise in (0.0, 0.4):
            key = f"{policy}|{noise}"
            res[key] = {w: run(w, p, policy, noise) for w, p in WORLDS.items()}
            mean = {b: statistics.mean(v[b] for v in res[key].values()) for b in ("manual-rule", "cbr-only", "cbr+manual")}
            print(key, {b: round(100 * m, 1) for b, m in mean.items()},
                  {w: {b: round(100 * x, 1) for b, x in v.items()} for w, v in res[key].items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
