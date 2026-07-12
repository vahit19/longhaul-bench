"""Per-process deployment-footprint measurement (council blocking item #6).

Reports RSS separately for each llama-server instance (identified by --port in
its command line) and each project python process — the artifact that backs
(and continuously audits) the README deployment table. With --watch, samples
over time and appends JSONL, enabling long-horizon runtime-growth measurement
with process-level attribution (replacing the retracted aggregate claim).

Usage:
    python scripts/footprint.py                     # one snapshot, table to stdout
    python scripts/footprint.py --watch 3600 --interval 60 --out runs/footprint_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import psutil


def snapshot() -> dict:
    procs = []
    for p in psutil.process_iter(["name", "memory_info", "cmdline", "create_time"]):
        try:
            name = p.info["name"] or ""
            cmd = p.info["cmdline"] or []
            if "llama-server" in name:
                port = next((cmd[i + 1] for i, c in enumerate(cmd) if c == "--port" and i + 1 < len(cmd)), "?")
                role = {"8080": "slm-server", "8081": "embed-server", "8082": "second-family-server"}.get(port, f"llama:{port}")
            elif "python" in name.lower() and any("longhaul" in (c or "") for c in cmd):
                role = "agent/orchestrator"
            else:
                continue
            procs.append({
                "role": role, "pid": p.pid,
                "rss_mb": round(p.info["memory_info"].rss / 1e6),
                "uptime_h": round((time.time() - p.info["create_time"]) / 3600, 2),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "processes": procs,
            "runtime_total_mb": sum(x["rss_mb"] for x in procs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", type=int, default=0, help="total seconds to sample (0 = single snapshot)")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    end = time.time() + args.watch
    while True:
        s = snapshot()
        line = json.dumps(s)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line, flush=True)
        if time.time() >= end:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
