"""Detached chainer: wait for w45 to finish, then run the w46 matrix."""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SENTINEL = REPO / "runs/repl_w45/framing-unverified/summary.json"

while not SENTINEL.exists():
    time.sleep(60)

subprocess.run([
    sys.executable, str(REPO / "scripts/run_matrix.py"),
    "--config", str(REPO / "runs/repl_config_full.json"),
    "--name", "repl_w46", "--limit", "945",
    "--probe-every", "100", "--probe-size", "50",
    "--world", str(REPO / "runs/repl_worlds/w46/world.json"),
    "--episodes", str(REPO / "runs/repl_worlds/w46/episodes.jsonl"),
], cwd=REPO, check=True)
