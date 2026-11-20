"""Incomplete-manual study extended from 1 to 5 plants.

Worlds g43-g46 are generated with the same generator and manual-gap rate (0.3) as the
original gap world (seed 47, runs/gap_world, reproduced exactly by
`environments/generator.py --seed 47 --episodes 1000 --manual-gap 0.3`):

    python environments/generator.py --seed <s> --episodes 1000 --manual-gap 0.3 --out runs/gap_worlds/g<s>

Arms and settings are those of the original study (runs/config_gap.json: frozen, reflect,
reflect with both consistency safeguards; Qwen2.5-3B Q4_K_M, canonical prompt, 945 episodes
after 50 probe episodes, feedback seed 7). The added plants were served on a GPU host with
the same llama.cpp build (b9918) and a byte-identical GGUF file (SHA-256 checked).

Usage: python scripts/chain_gap5.py <endpoint> <arm-config.json> <seed>
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.pop("LONGHAUL_STRICT_VOCAB", None)
if not sys.argv[1].startswith("http://127.0.0.1"):
    os.environ["LONGHAUL_REMOTE_SERVER"] = "1"
endpoint, config, seed = sys.argv[1], sys.argv[2], sys.argv[3]
wdir = REPO / "runs" / "gap_worlds" / f"g{seed}"
subprocess.run([
    sys.executable, str(REPO / "scripts/run_matrix.py"),
    "--config", str(REPO / config), "--name", f"gap_study_g{seed}", "--limit", "945",
    "--probe-every", "200", "--probe-size", "50", "--endpoint", endpoint,
    "--world", str(wdir / "world.json"), "--episodes", str(wdir / "episodes.jsonl"),
], cwd=REPO, check=True)
