"""Experiment-matrix orchestrator for LongHaul-Bench.

Runs agents/longrun.py once per configuration (sequentially — the SLM server
is the bottleneck), collects every summary.json, and writes one aggregate
markdown table + CSV. Configurations come from a JSON file or the built-in
smoke matrix.

Usage:
    local/venv/Scripts/python scripts/run_matrix.py --limit 10 --name m3_matrix_smoke
    local/venv/Scripts/python scripts/run_matrix.py --config runs/full_matrix.json --name m4_full
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SMOKE_MATRIX = [
    {"operator": "frozen"},
    {"operator": "append", "policy": "fifo"},
    {"operator": "reflect", "policy": "compress"},
    {"operator": "gated", "policy": "importance"},
    {"operator": "frozen", "oracle": True, "tag": "oracle"},
    {"operator": "reflect", "policy": "compress", "feedback_noise": 0.4, "tag": "noise0.4"},
]

COLUMNS = ["run", "operator", "policy", "budget", "feedback_noise", "episodes",
           "exact_accuracy", "memory_hit_rate", "latency_p50_s", "peak_system_rss_mb"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, help="JSON list of run configs (default: built-in smoke matrix)")
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--probe-every", type=int, default=0, help="0 = limit//2")
    p.add_argument("--probe-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--world", default="runs/v01/world.json")
    p.add_argument("--episodes", default="runs/v01/episodes.jsonl")
    args = p.parse_args()

    matrix = json.loads(args.config.read_text(encoding="utf-8")) if args.config else SMOKE_MATRIX
    base = REPO / "runs" / args.name
    rows = []

    for cfg in matrix:
        tag = cfg.get("tag") or cfg["operator"]
        out = base / tag
        cmd = [sys.executable, str(REPO / "agents" / "longrun.py"),
               "--world", args.world, "--episodes", args.episodes,
               "--operator", cfg["operator"],
               "--policy", cfg.get("policy", "fifo"),
               "--budget", str(cfg.get("budget", 100)),
               "--feedback-noise", str(cfg.get("feedback_noise", 0.0)),
               "--limit", str(args.limit), "--seed", str(args.seed),
               "--probe-every", str(args.probe_every or max(args.limit // 2, 1)),
               "--probe-size", str(args.probe_size),
               "--traces", "--out", str(out)]
        if cfg.get("oracle"):
            cmd.append("--oracle")
        if cfg.get("defense"):
            cmd += ["--defense", cfg["defense"]]
        if cfg.get("memory_label"):
            cmd += ["--memory-label", cfg["memory_label"]]
        print(f"=== {tag} ===", flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        rows.append({"run": tag, **{c: s.get(c, "") for c in COLUMNS[1:]}})

    with (base / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    md = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"
    for r in rows:
        md += "| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |\n"
    (base / "matrix.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
