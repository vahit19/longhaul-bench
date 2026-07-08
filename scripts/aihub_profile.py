"""Qualcomm AI Hub profiling integration for LongHaul-Bench (M5).

Submits a model to Qualcomm AI Hub and retrieves on-REAL-device latency and
memory metrics (Snapdragon phones/laptops, and industrial IoT boards such as
RB3 Gen 2 — the most relevant target for this benchmark).

One-time setup (account owner must do this — requires a Qualcomm ID):
    1. Create an account at https://aihub.qualcomm.com
    2. Get the API token from account settings
    3. local/venv/Scripts/qai-hub configure --api_token <TOKEN>

Usage:
    local/venv/Scripts/python scripts/aihub_profile.py --list-devices
    local/venv/Scripts/python scripts/aihub_profile.py --model model.onnx --device "QCS6490 (Proxy)"

Note: the Qwen chat model goes to device via the GenieX llama.cpp plugin path
(documented at https://aihub.qualcomm.com — LLM recipes), not via this ONNX
profiling flow; this script covers the embedding model and any ONNX-exported
components, plus device latency/memory ground truth for the paper's M5 table.

STATUS: ready but UNTESTED — blocked on an API token (account owner action).
"""

from __future__ import annotations

import argparse

import qai_hub as hub


def list_devices() -> None:
    for d in hub.get_devices():
        attrs = " ".join(d.attributes)
        if any(k in attrs for k in ("iot", "industrial", "proxy")) or "QCS" in d.name:
            marker = "  <-- industrial/IoT class"
        else:
            marker = ""
        print(f"{d.name:45s} os={d.os}{marker}")


def profile(model_path: str, device_name: str) -> None:
    device = hub.Device(device_name)
    job = hub.submit_profile_job(model=model_path, device=device)
    profile_data = job.download_profile()
    summary = profile_data.get("execution_summary", {})
    print(f"device:            {device_name}")
    print(f"inference p50:     {summary.get('estimated_inference_time', '?')} us")
    print(f"peak memory:       {summary.get('estimated_inference_peak_memory', '?')} bytes")
    print(f"full job details:  {job.url}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--model")
    p.add_argument("--device", default="QCS6490 (Proxy)")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
    elif args.model:
        profile(args.model, args.device)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
