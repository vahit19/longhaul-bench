"""Reconstruct results.jsonl + summary.json from raw traces.

Traces (traces.jsonl) are the primary raw record: the full message log of every
episode, including the agent's final diagnosis. When a derived results file is
missing, this script rebuilds it deterministically by re-scoring the traced
diagnosis against the (seed-regenerable) episode ground truth.

Reconstructed files carry "reconstructed_from_traces": true and omit
latency/token fields (those were runtime measurements, not derivable from
traces; accuracy and memory-usage fields are fully derivable).

Usage:
    python scripts/reconstruct_results.py --arm runs/repl_w43/frozen \
        --episodes runs/repl_worlds/w43/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def last_diagnosis(trace: list) -> dict:
    """The agent's final committed diagnosis in the message log."""
    dec = json.JSONDecoder()
    for msg in reversed(trace):
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content", "")
        i = text.find("{")
        while i != -1:
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict) and "diagnosis" in obj:
                    return obj["diagnosis"] or {}
            except json.JSONDecodeError:
                pass
            i = text.find("{", i + 1)
    return {}


def reconstruct(arm: Path, episodes_path: Path) -> dict:
    episodes = {e["episode_id"]: e for e in map(json.loads, episodes_path.open(encoding="utf-8"))}
    rows = []
    with (arm / "traces.jsonl").open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            ep = episodes.get(t["episode_id"])
            if ep is None:
                continue
            d = last_diagnosis(t["trace"])
            gt = ep["ground_truth"]
            user_msg = next((m["content"] for m in t["trace"] if m["role"] == "user"), "")
            rows.append({
                "episode_id": t["episode_id"],
                "status": "ok" if d else "no_diagnosis",
                "predicted": {"component": d.get("component", ""), "failure_mode": d.get("failure_mode", "")},
                "component_correct": d.get("component") == gt["component"],
                "exact_correct": d.get("component") == gt["component"] and d.get("failure_mode") == gt["failure_mode"],
                "memory_used": ("Past confirmed cases" in user_msg) or ("Past case notes" in user_msg),
                "reconstructed_from_traces": True,
            })
    n = len(rows)
    summary = {
        "episodes": n,
        "exact_accuracy": sum(r["exact_correct"] for r in rows) / n if n else None,
        "component_accuracy": sum(r["component_correct"] for r in rows) / n if n else None,
        "memory_hit_rate": sum(r["memory_used"] for r in rows) / n if n else None,
        "no_diagnosis_rate": sum(r["status"] == "no_diagnosis" for r in rows) / n if n else None,
        "reconstructed_from_traces": True,
        "note": "results rebuilt from raw traces; runtime fields (latency/tokens) not derivable and omitted",
    }
    with (arm / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (arm / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    args = p.parse_args()
    s = reconstruct(args.arm, args.episodes)
    print(json.dumps({str(args.arm): s}, indent=2))


if __name__ == "__main__":
    main()
