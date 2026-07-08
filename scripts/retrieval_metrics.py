"""Deterministic retrieval-quality metrics (no LLM judge needed).

The environment knows each episode's true cause, so retrieval quality is
computed exactly: a manual row is RELEVANT iff its possible_causes contain
the ground-truth "component (mode)". Reports hit@k and MRR for keyword and
vector retrieval on identical queries.

Usage (embed server on :8081 required for vector mode):
    local/venv/Scripts/python scripts/retrieval_metrics.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.slm_agent import Tools  # noqa: E402


def evaluate_mode(episodes: list, world: dict, alarm_symptom: dict, retriever, k: int = 5) -> dict:
    hits, rr = 0, []
    for ep in episodes:
        symptoms = [alarm_symptom.get(a["code"], "") for a in ep["alarms"]]
        query = "diagnose fault with symptoms: " + ", ".join(s for s in symptoms if s)
        rows = Tools(world, ep["machine_id"], ep["alarms"], retriever=retriever).manual_search(query=query)[:k]
        gt = f"{ep['ground_truth']['component']} ({ep['ground_truth']['failure_mode']})"
        rank = next((i + 1 for i, r in enumerate(rows) if gt in r["possible_causes"]), None)
        hits += rank is not None
        rr.append(1 / rank if rank else 0.0)
    return {"hit@5": round(hits / len(episodes), 3), "mrr": round(sum(rr) / len(rr), 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()

    world = json.loads((REPO / "runs/v01/world.json").read_text(encoding="utf-8"))
    alarm_symptom = {a["code"]: a["symptom"] for a in world["alarm_table"]}
    episodes = [json.loads(l) for l in (REPO / "runs/v01/episodes.jsonl").open(encoding="utf-8")][: args.limit]

    from environments.retrieval import VectorIndex
    report = {
        "episodes": len(episodes),
        "keyword": evaluate_mode(episodes, world, alarm_symptom, retriever=None),
        "vector": evaluate_mode(episodes, world, alarm_symptom, retriever=VectorIndex(world)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
