"""Quantized-SLM diagnostic agent for LongHaul-Bench (M2, first cut).

A tool-using agent driven by a local llama.cpp server (OpenAI-compatible
API). The model sees an episode (alarms, log excerpt, operator note) and may
call tools — alarm_lookup, manual_search, maintenance_history — before
committing to a diagnosis. Strict JSON protocol; one structured retry on
malformed output, then the episode is scored as a failure (anomaly policy:
nothing is silently dropped).

Measures per episode: wall latency, prompt/completion tokens, tool calls.

Usage:
    local/bin/llama-server.exe -m local/models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8080
    python agents/slm_agent.py --world runs/v01/world.json --episodes runs/v01/episodes.jsonl --limit 100 --out runs/m2_smoke
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """You are an industrial maintenance diagnostic agent running on-site.
Diagnose the root cause of the reported machine problem.

You respond ONLY with a single JSON object, no other text. Exactly ONE action
per turn: one tool call OR one final diagnosis — never several. After each tool
call you will receive the result and can decide your next single action.
Two forms are allowed:

1. Call a tool:
{"tool": "manual_search", "args": {"query": "<symptom or keyword>"}}
{"tool": "alarm_lookup", "args": {}}
{"tool": "maintenance_history", "args": {}}

2. Commit to a final diagnosis (only when confident):
{"diagnosis": {"component": "<component_name>", "failure_mode": "<mode_name>"}}

Rules: use tools before diagnosing; alarm_lookup decodes the active alarm codes;
manual_search returns troubleshooting rows for this machine; component and
failure_mode must be values seen in tool results (e.g. "bearing", "wear")."""

# Optional strict-vocabulary clause (env LONGHAUL_STRICT_VOCAB=1). Default OFF so
# the canonical Qwen results keep their exact prompt; used for the cross-family probe.
import os as _os  # noqa: E402
if _os.environ.get("LONGHAUL_STRICT_VOCAB") == "1":
    SYSTEM_PROMPT += """

CRITICAL: manual_search returns rows whose "possible_causes" look like
"electric_motor (winding_fault)", "bearing (wear)". Your failure_mode MUST be
copied EXACTLY from inside those parentheses (e.g. "winding_fault", not "wear"
or "malfunction"), and component MUST be the name before the parenthesis. Never
invent a mode word; only use ones that appear in a tool result for this machine."""


class Tools:
    def __init__(self, world: dict, machine_id: str, alarms: list, retriever=None):
        self.machine_id = machine_id
        self.alarms = alarms
        self.retriever = retriever  # optional VectorIndex; None -> keyword search
        self.alarm_symptom = {a["code"]: a["symptom"] for a in world["alarm_table"]}
        self.manual_rows = [
            row
            for m in world["manuals"] if m["machine_id"] == machine_id
            for s in m["sections"] for row in s["rows"]
        ]
        self.history = [h for h in world["maintenance_history"] if h["machine_id"] == machine_id]

    def alarm_lookup(self, **_) -> list:
        return [
            {"code": a["code"], "symptom": self.alarm_symptom.get(a["code"], "unknown")}
            for a in self.alarms
        ]

    def manual_search(self, query: str = "", **_) -> list:
        if self.retriever is not None:
            return self.retriever.search(self.machine_id, query or "fault diagnosis", k=5)
        words = set(re.findall(r"[a-z_]{3,}", query.lower()))
        scored = []
        for row in self.manual_rows:
            text = (row["symptom"] + " " + " ".join(row["possible_causes"])).lower()
            score = sum(1 for w in words if w in text)
            scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        return [r for s, r in scored[:5] if s > 0] or [r for _, r in scored[:5]]

    def maintenance_history(self, **_) -> list:
        return self.history[-5:]


def chat(endpoint: str, messages: list, max_tokens: int = 200, retries: int = 3) -> tuple:
    payload = json.dumps({
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                endpoint + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                body = json.loads(r.read())
            usage = body.get("usage", {})
            return body["choices"][0]["message"]["content"], usage
        except (TimeoutError, OSError) as e:  # transient server hiccups on long runs
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise last_err


def parse_json(text: str):
    """Return the FIRST valid JSON object in the text (small models often
    emit several actions at once; we execute one action per turn)."""
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return None


def run_episode(endpoint: str, world: dict, ep: dict, max_steps: int = 6,
                memory_context: str = "", retriever=None, include_trace: bool = False) -> dict:
    tools = Tools(world, ep["machine_id"], ep["alarms"], retriever=retriever)
    user = (
        f"Machine {ep['machine_id']} problem report.\n"
        f"Active alarms: {json.dumps(ep['alarms'])}\n"
        f"Log excerpt: {json.dumps(ep['log_excerpt'])}\n"
        f"Operator note: {ep['operator_note']}"
    )
    if memory_context:
        user += "\n\n" + memory_context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    t0 = time.time()
    tokens = {"prompt": 0, "completion": 0}
    tool_calls, anomalies = 0, 0

    for _ in range(max_steps):
        text, usage = chat(endpoint, messages)
        tokens["prompt"] += usage.get("prompt_tokens", 0)
        tokens["completion"] += usage.get("completion_tokens", 0)
        obj = parse_json(text)

        if obj is None:  # anomaly policy: one structured retry, then fail
            anomalies += 1
            if anomalies > 1:
                break
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Invalid output. Respond with ONE valid JSON object only."})
            continue

        if "diagnosis" in obj:
            d = obj["diagnosis"] or {}
            r = _result(ep, d.get("component", ""), d.get("failure_mode", ""),
                        t0, tokens, tool_calls, anomalies, "ok")
            if include_trace:
                r["trace"] = messages + [{"role": "assistant", "content": text}]
            return r

        if "tool" in obj and hasattr(tools, str(obj["tool"])):
            tool_calls += 1
            result = getattr(tools, obj["tool"])(**(obj.get("args") or {}))
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
        else:
            anomalies += 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Unknown tool. Use alarm_lookup, manual_search, maintenance_history, or give a diagnosis."})

    r = _result(ep, "", "", t0, tokens, tool_calls, anomalies, "no_diagnosis")
    if include_trace:
        r["trace"] = messages
    return r


def _result(ep, component, mode, t0, tokens, tool_calls, anomalies, status) -> dict:
    gt = ep["ground_truth"]
    return {
        "episode_id": ep["episode_id"],
        "status": status,
        "predicted": {"component": component, "failure_mode": mode},
        "component_correct": component == gt["component"],
        "exact_correct": component == gt["component"] and mode == gt["failure_mode"],
        "latency_s": round(time.time() - t0, 2),
        "tokens": tokens,
        "tool_calls": tool_calls,
        "anomalies": anomalies,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--out", type=Path, default=Path("runs/m2_smoke"))
    args = p.parse_args()

    world = json.loads(args.world.read_text(encoding="utf-8"))
    episodes = [json.loads(l) for l in args.episodes.open(encoding="utf-8")][: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as f:
        for i, ep in enumerate(episodes, 1):
            r = run_episode(args.endpoint, world, ep)
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(episodes):
                acc = sum(x["exact_correct"] for x in results) / len(results)
                print(f"[{i}/{len(episodes)}] exact={acc:.1%} "
                      f"median_latency={statistics.median(x['latency_s'] for x in results):.1f}s", flush=True)

    lat = sorted(x["latency_s"] for x in results)
    summary = {
        "episodes": len(results),
        "component_accuracy": sum(x["component_correct"] for x in results) / len(results),
        "exact_accuracy": sum(x["exact_correct"] for x in results) / len(results),
        "latency_p50_s": lat[len(lat) // 2],
        "latency_p95_s": lat[int(len(lat) * 0.95) - 1],
        "mean_tokens_per_episode": round(statistics.mean(
            x["tokens"]["prompt"] + x["tokens"]["completion"] for x in results)),
        "mean_tool_calls": round(statistics.mean(x["tool_calls"] for x in results), 2),
        "anomaly_rate": sum(x["anomalies"] > 0 for x in results) / len(results),
        "no_diagnosis_rate": sum(x["status"] == "no_diagnosis" for x in results) / len(results),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
