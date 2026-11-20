"""Third model family at edge scale (Gemma-3-4B), one arm on one world.

The model is served through the OpenRouter API with one pinned provider, temperature 0
and a fixed seed; the canonical prompt, worlds, episode streams, feedback seed and
episode count are the same as in the main Qwen runs. The key is read from the
git-ignored .env file and is never written to logs.

Usage: python scripts/chain_api.py <world> <arm-config.json>
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL, PROVIDER, TAG = "google/gemma-3-4b-it", "Parasail", "gemma4b"
# Same episode stream per world as the main Qwen runs: worlds 42-44 skip 5 probe
# episodes and run 995, worlds 45-46 skip 50 and run 945. Probes are not re-run
# (they never write to the case base and only feed the unreported probe curve).
WORLDS = {
    "w42": ("data/longhaul-v0-standard/world.json", "data/longhaul-v0-standard/episodes.jsonl", 5, 995),
    **{w: (f"runs/repl_worlds/{w}/world.json", f"runs/repl_worlds/{w}/episodes.jsonl", 5, 995)
       for w in ("w43", "w44")},
    **{w: (f"runs/repl_worlds/{w}/world.json", f"runs/repl_worlds/{w}/episodes.jsonl", 50, 945)
       for w in ("w45", "w46")},
}

for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("OPENROUTER_API_KEY="):
        os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
os.environ["LONGHAUL_API_MODEL"] = MODEL
os.environ["LONGHAUL_API_PROVIDER"] = PROVIDER
os.environ["LONGHAUL_REMOTE_SERVER"] = "1"
os.environ.pop("LONGHAUL_STRICT_VOCAB", None)  # canonical prompt, as in the main Qwen runs

world, config = sys.argv[1], sys.argv[2]
wpath, epath, probe_size, limit = WORLDS[world]
subprocess.run([
    sys.executable, str(REPO / "scripts/run_matrix.py"),
    "--config", str(REPO / config),
    "--name", f"api_{TAG}_{world}", "--limit", str(int(sys.argv[3]) if len(sys.argv) > 3 else limit),
    "--probe-every", "1000000", "--probe-size", str(probe_size), "--endpoint", "https://openrouter.ai/api",
    "--world", str(REPO / wpath), "--episodes", str(REPO / epath),
], cwd=REPO, check=True)
