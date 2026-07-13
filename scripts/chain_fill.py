"""Detached chainer: fill the 3 missing arms (oracle/gated/noise0.2) on w43 and
w44, using each world's ORIGINAL config (limit 995, probe-size 5) so every
world stays internally consistent. Resume support skips the 6 existing arms.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for world in ("w43", "w44"):
    subprocess.run([
        sys.executable, str(REPO / "scripts/run_matrix.py"),
        "--config", str(REPO / "runs/repl_config_full.json"),
        "--name", f"repl_{world}", "--limit", "995",
        "--probe-every", "50", "--probe-size", "5",
        "--world", str(REPO / f"runs/repl_worlds/{world}/world.json"),
        "--episodes", str(REPO / f"runs/repl_worlds/{world}/episodes.jsonl"),
    ], cwd=REPO, check=True)
