"""Heuristic baseline agent for LongHaul-Bench.

A deliberately simple, LLM-free diagnostician: it looks up alarm codes,
collects candidate causes from the machine manual, and disambiguates using
keyword overlap between the observed log excerpt and each failure mode's
known log signature. This is the floor every learned agent must beat —
and an end-to-end test of the environment pipeline.

Usage:
    python environments/generator.py --machines 5 --episodes 1000 --seed 42 --out runs/v01
    python agents/baseline.py --world runs/v01/world.json --episodes runs/v01/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.generator import FAILURE_MODES  # noqa: E402

_WORD = re.compile(r"[a-z]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def diagnose(episode: dict, alarm_symptom: dict[str, str], manual_rows: dict[str, list[dict]]) -> dict:
    """Return the agent's diagnosis for one episode."""
    symptoms = [alarm_symptom[a["code"]] for a in episode["alarms"] if a["code"] in alarm_symptom]

    # 1) Candidate causes: manual rows for each observed symptom.
    votes: Counter[str] = Counter()
    for row in manual_rows.get(episode["machine_id"], []):
        if row["symptom"] in symptoms:
            for cause in row["possible_causes"]:
                votes[cause] += 1

    if not votes:
        return {"component": "unknown", "failure_mode": "unknown"}

    # 2) Disambiguate: keyword overlap between observed log and each
    #    candidate mode's log signature (from the failure-mode knowledge base).
    log_words = _words(" ".join(episode["log_excerpt"]))
    best, best_score = None, (-1, -1.0)
    for cause, vote_count in votes.items():
        component, mode = re.match(r"(.+) \((.+)\)", cause).groups()
        signature = FAILURE_MODES[component][mode]["log"]
        overlap = len(log_words & _words(signature)) / max(len(_words(signature)), 1)
        score = (vote_count, overlap)
        if score > best_score:
            best_score, best = score, (component, mode)

    return {"component": best[0], "failure_mode": best[1]}


def main() -> None:
    p = argparse.ArgumentParser(description="Run the heuristic baseline over an episode stream.")
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    args = p.parse_args()

    world = json.loads(args.world.read_text(encoding="utf-8"))
    alarm_symptom = {a["code"]: a["symptom"] for a in world["alarm_table"]}
    manual_rows = {
        m["machine_id"]: [row for s in m["sections"] for row in s["rows"]]
        for m in world["manuals"]
    }

    total = component_ok = exact_ok = 0
    with args.episodes.open(encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            d = diagnose(ep, alarm_symptom, manual_rows)
            gt = ep["ground_truth"]
            total += 1
            component_ok += d["component"] == gt["component"]
            exact_ok += d["component"] == gt["component"] and d["failure_mode"] == gt["failure_mode"]

    print(f"episodes:            {total}")
    print(f"component accuracy:  {component_ok / total:.1%}")
    print(f"exact accuracy:      {exact_ok / total:.1%}  (component + failure mode)")


if __name__ == "__main__":
    main()
