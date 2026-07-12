"""Knowledge-state autopsy with episode-matched controls (council blocking item #2).

Reproduces the recall-quality split for a memory arm AND — critically — computes
the frozen-control accuracy on the SAME episode IDs. Because the episode stream
is deterministic, this is a paired comparison that separates the two rival
explanations for the misleading-recall collapse:

  H_memory: misleading recall CAUSES errors -> frozen (no memory) should do
            roughly its average on those same episodes.
  H_difficulty: misleading-recall episodes are intrinsically harder -> frozen
            should collapse on them too.

Usage:
    python scripts/autopsy.py --memory-arm runs/m4_night1/append --frozen-arm runs/m4_night1/frozen \
        --episodes data/longhaul-v0-standard/episodes.jsonl --out docs/paper/autopsy_matched.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(arm: Path) -> dict:
    return {r["episode_id"]: r for r in map(json.loads, (arm / "results.jsonl").open(encoding="utf-8"))}


def classify_episodes(arm: Path, episodes: dict) -> dict:
    """Split episode IDs by memory-context quality in the arm's traces."""
    groups = {"correct_hint": [], "misleading_hint": [], "no_memory": []}
    for line in (arm / "traces.jsonl").open(encoding="utf-8"):
        t = json.loads(line)
        ep = episodes.get(t["episode_id"])
        if ep is None:
            continue
        user_msg = next((m["content"] for m in t["trace"] if m["role"] == "user"), "")
        idx = user_msg.find("Past confirmed cases")
        if idx < 0:
            idx = user_msg.find("Past case notes")
        if idx < 0:
            groups["no_memory"].append(t["episode_id"])
            continue
        block = user_msg[idx:]
        gt = f"{ep['ground_truth']['component']} ({ep['ground_truth']['failure_mode']})"
        groups["correct_hint" if gt in block else "misleading_hint"].append(t["episode_id"])
    return groups


def acc(results: dict, ids: list) -> float:
    hits = [results[i]["exact_correct"] for i in ids if i in results]
    return sum(hits) / len(hits) if hits else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-arm", type=Path, required=True)
    p.add_argument("--frozen-arm", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    episodes = {e["episode_id"]: e for e in map(json.loads, args.episodes.open(encoding="utf-8"))}
    mem_res = load_results(args.memory_arm)
    frz_res = load_results(args.frozen_arm)
    groups = classify_episodes(args.memory_arm, episodes)

    report = {"memory_arm": str(args.memory_arm), "frozen_arm": str(args.frozen_arm), "groups": {}}
    for g, ids in groups.items():
        report["groups"][g] = {
            "n": len(ids),
            "memory_arm_accuracy": round(acc(mem_res, ids), 4),
            "frozen_matched_accuracy": round(acc(frz_res, ids), 4),
        }
    d = report["groups"]
    if d["misleading_hint"]["n"]:
        gap_mem = d["correct_hint"]["memory_arm_accuracy"] - d["misleading_hint"]["memory_arm_accuracy"]
        gap_frz = d["correct_hint"]["frozen_matched_accuracy"] - d["misleading_hint"]["frozen_matched_accuracy"]
        report["difficulty_share_of_gap"] = round(gap_frz / gap_mem, 3) if gap_mem else None
        report["memory_caused_delta_on_misleading"] = round(
            d["misleading_hint"]["memory_arm_accuracy"] - d["misleading_hint"]["frozen_matched_accuracy"], 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
