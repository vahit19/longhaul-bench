"""Run one arm on one of the five main plants with the episode stream and probe cadence of the
main Qwen study (seed 42 and plants 43-44: 5 probe episodes, probes every 50, 995 episodes;
plants 45-46: 50 probe episodes, probes every 100, 945 episodes; feedback seed 7). Used for
(a) support-aware eviction run with the agent and (b) Gemma-3-4B Q4_K_M, both served by the
same llama.cpp build (b9918) on a GPU host.

Usage: python scripts/chain_main5.py <name> <endpoint> <arm-config.json> <world> [--strict]
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORLDS = {
    "w42": ("data/longhaul-v0-standard", 5, 50, 995),
    "w43": ("runs/repl_worlds/w43", 5, 50, 995),
    "w44": ("runs/repl_worlds/w44", 5, 50, 995),
    "w45": ("runs/repl_worlds/w45", 50, 100, 945),
    "w46": ("runs/repl_worlds/w46", 50, 100, 945),
}
name, endpoint, config, world = sys.argv[1:5]
if "--strict" in sys.argv[5:]:
    os.environ["LONGHAUL_STRICT_VOCAB"] = "1"
else:
    os.environ.pop("LONGHAUL_STRICT_VOCAB", None)
if not endpoint.startswith("http://127.0.0.1"):
    os.environ["LONGHAUL_REMOTE_SERVER"] = "1"
wdir, probe_size, probe_every, limit = WORLDS[world]
subprocess.run([
    sys.executable, str(REPO / "scripts/run_matrix.py"),
    "--config", str(REPO / config), "--name", f"{name}_{world}", "--limit", str(limit),
    "--probe-every", str(probe_every), "--probe-size", str(probe_size), "--endpoint", endpoint,
    "--world", str(REPO / wdir / "world.json"), "--episodes", str(REPO / wdir / "episodes.jsonl"),
], cwd=REPO, check=True)
