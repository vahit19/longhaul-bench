"""LangGraph comparison arm for LongHaul-Bench.

The same diagnostic protocol as agents/slm_agent.py, expressed as a LangGraph
StateGraph: an `agent` node (local SLM via ChatOpenAI -> llama-server) and a
`tools` node connected by a conditional edge. Purpose: (a) demonstrate the
benchmark under a mainstream orchestration framework, (b) measure framework
overhead vs the dependency-light bare loop on identical episodes.

Usage (llama-server on :8080 required; run with local/venv python):
    local/venv/Scripts/python agents/langgraph_agent.py \
        --world runs/v01/world.json --episodes runs/v01/episodes.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.slm_agent import SYSTEM_PROMPT, Tools, parse_json  # noqa: E402

MAX_STEPS = 6


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    diagnosis: dict
    steps: int
    anomalies: int
    tool_calls: int


def build_graph(llm: ChatOpenAI, tools: Tools):
    def agent_node(state: AgentState) -> dict:
        reply: AIMessage = llm.invoke(state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def route(state: AgentState) -> str:
        if state["steps"] >= MAX_STEPS or state["anomalies"] > 1:
            return END
        obj = parse_json(state["messages"][-1].content)
        if obj is None or ("tool" not in obj and "diagnosis" not in obj):
            return "malformed"
        return "final" if "diagnosis" in obj else "tools"

    def tools_node(state: AgentState) -> dict:
        obj = parse_json(state["messages"][-1].content)
        if hasattr(tools, str(obj["tool"])):
            result = getattr(tools, obj["tool"])(**(obj.get("args") or {}))
            msg = f"Tool result: {json.dumps(result)}"
            extra = {"tool_calls": state["tool_calls"] + 1}
        else:
            msg = "Unknown tool. Use alarm_lookup, manual_search, maintenance_history, or give a diagnosis."
            extra = {"anomalies": state["anomalies"] + 1}
        return {"messages": [HumanMessage(content=msg)], **extra}

    def malformed_node(state: AgentState) -> dict:
        return {
            "messages": [HumanMessage(content="Invalid output. Respond with ONE valid JSON object only.")],
            "anomalies": state["anomalies"] + 1,
        }

    def final_node(state: AgentState) -> dict:
        obj = parse_json(state["messages"][-1].content)
        return {"diagnosis": (obj or {}).get("diagnosis") or {}}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("malformed", malformed_node)
    g.add_node("final", final_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "malformed": "malformed", "final": "final", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("malformed", "agent")
    g.add_edge("final", END)
    return g.compile()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    args = p.parse_args()

    world = json.loads(args.world.read_text(encoding="utf-8"))
    episodes = [json.loads(l) for l in args.episodes.open(encoding="utf-8")][: args.limit]
    llm = ChatOpenAI(base_url=args.endpoint, api_key="sk-local", model="qwen2.5-3b-instruct", temperature=0)

    results = []
    for ep in episodes:
        tools = Tools(world, ep["machine_id"], ep["alarms"])
        graph = build_graph(llm, tools)
        user = (
            f"Machine {ep['machine_id']} problem report.\n"
            f"Active alarms: {json.dumps(ep['alarms'])}\n"
            f"Log excerpt: {json.dumps(ep['log_excerpt'])}\n"
            f"Operator note: {ep['operator_note']}"
        )
        t0 = time.time()
        out = graph.invoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)],
             "diagnosis": {}, "steps": 0, "anomalies": 0, "tool_calls": 0},
            {"recursion_limit": 4 * MAX_STEPS},
        )
        gt = ep["ground_truth"]
        d = out["diagnosis"]
        results.append({
            "exact": d.get("component") == gt["component"] and d.get("failure_mode") == gt["failure_mode"],
            "latency_s": round(time.time() - t0, 2),
            "tool_calls": out["tool_calls"],
        })
        print(f"{ep['episode_id']}: exact={results[-1]['exact']} latency={results[-1]['latency_s']}s")

    lat = sorted(r["latency_s"] for r in results)
    print(json.dumps({
        "framework": "langgraph",
        "episodes": len(results),
        "exact_accuracy": sum(r["exact"] for r in results) / len(results),
        "latency_p50_s": lat[len(lat) // 2],
        "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in results), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
