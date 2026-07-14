"""Cross-family robustness: run core dynamic arms (frozen/reflect/rot/defense)
for BOTH Qwen (8080) and Llama-3.2 (8082) under the SAME strict-vocab prompt,
on 2 worlds. Shows whether learning/rot/defense DIRECTIONS replicate across
model families under an identical prompt. LONGHAUL_STRICT_VOCAB set by launcher.
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["LONGHAUL_STRICT_VOCAB"] = "1"  # guaranteed for all child subprocesses
REPO = Path(__file__).resolve().parents[1]
MODELS = [("qwen", "http://127.0.0.1:8080"), ("llama", "http://127.0.0.1:8082")]
WORLDS = ["w45", "w46"]

for model, endpoint in MODELS:
    for world in WORLDS:
        subprocess.run([
            sys.executable, str(REPO / "scripts/run_matrix.py"),
            "--config", str(REPO / "runs/config_core.json"),
            "--name", f"xfam_{model}_{world}", "--limit", "945",
            "--probe-every", "200", "--probe-size", "50", "--endpoint", endpoint,
            "--world", str(REPO / f"runs/repl_worlds/{world}/world.json"),
            "--episodes", str(REPO / f"runs/repl_worlds/{world}/episodes.jsonl"),
        ], cwd=REPO, check=True)
