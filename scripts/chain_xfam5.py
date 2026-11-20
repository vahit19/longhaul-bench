"""Cross-family robustness extended from 2 to 5 worlds.

Runs the core dynamic arms on the three worlds not covered by chain_xfam.py
(seed 42 = data/longhaul-v0-standard, w43, w44) under the same strict-vocabulary prompt and the same
settings as chain_xfam.py (945 episodes, probe every 200, feedback seed 7).

The three added worlds were served on a GPU host with the same llama.cpp build (b9918)
and byte-identical GGUF files (SHA-256 checked) as the CPU runs.

Usage: python scripts/chain_xfam5.py <model> <endpoint> <config.json> [world]
  e.g. python scripts/chain_xfam5.py llama http://127.0.0.1:8082 runs/config_core.json
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["LONGHAUL_STRICT_VOCAB"] = "1"  # guaranteed for all child subprocesses
if not sys.argv[2].startswith("http://127.0.0.1"):
    os.environ["LONGHAUL_REMOTE_SERVER"] = "1"
REPO = Path(__file__).resolve().parents[1]
WORLDS = {
    "w42": ("data/longhaul-v0-standard/world.json", "data/longhaul-v0-standard/episodes.jsonl"),
    "w43": ("runs/repl_worlds/w43/world.json", "runs/repl_worlds/w43/episodes.jsonl"),
    "w44": ("runs/repl_worlds/w44/world.json", "runs/repl_worlds/w44/episodes.jsonl"),
}

model, endpoint, config = sys.argv[1], sys.argv[2], sys.argv[3]
selected = sys.argv[4:] or list(WORLDS)
for world, (wpath, epath) in ((w, WORLDS[w]) for w in selected):
    subprocess.run([
        sys.executable, str(REPO / "scripts/run_matrix.py"),
        "--config", str(REPO / config),
        "--name", f"xfam_{model}_{world}", "--limit", "945",
        "--probe-every", "200", "--probe-size", "50", "--endpoint", endpoint,
        "--world", str(REPO / wpath), "--episodes", str(REPO / epath),
    ], cwd=REPO, check=True)
