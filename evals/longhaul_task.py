"""Inspect AI task for LongHaul-Bench diagnostic episodes.

Wraps the benchmark in the Inspect evaluation framework (https://inspect.aisi.org.uk):
episodes become Samples, the tool-loop agent becomes a custom Solver, and
diagnosis matching becomes a Scorer with accuracy/stderr metrics. Run logs are
written in Inspect's standard format (viewable with `inspect view`).

Usage (llama-server must be running):
    set LOCAL_BASE_URL=http://127.0.0.1:8080/v1
    set LOCAL_API_KEY=sk-local
    local/venv/Scripts/inspect eval evals/longhaul_task.py \
        --model openai-api/local/qwen2.5-3b-instruct --limit 10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.slm_agent import SYSTEM_PROMPT, Tools, parse_json  # noqa: E402

WORLD = json.loads((REPO / "runs" / "v01" / "world.json").read_text(encoding="utf-8"))
EPISODES = REPO / "runs" / "v01" / "episodes.jsonl"

# retrieval mode: LONGHAUL_RETRIEVAL=vector -> Qdrant + embeddings; default keyword
import os  # noqa: E402

RETRIEVER = None
if os.environ.get("LONGHAUL_RETRIEVAL") == "vector":
    from environments.retrieval import VectorIndex  # noqa: E402
    RETRIEVER = VectorIndex(WORLD, os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8081"))


def episode_dataset() -> MemoryDataset:
    samples = []
    with EPISODES.open(encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            samples.append(
                Sample(
                    id=ep["episode_id"],
                    input=(
                        f"Machine {ep['machine_id']} problem report.\n"
                        f"Active alarms: {json.dumps(ep['alarms'])}\n"
                        f"Log excerpt: {json.dumps(ep['log_excerpt'])}\n"
                        f"Operator note: {ep['operator_note']}"
                    ),
                    target=json.dumps(ep["ground_truth"]),
                    metadata={"machine_id": ep["machine_id"], "alarms": ep["alarms"]},
                )
            )
    return MemoryDataset(samples)


@solver
def diagnostic_loop(max_steps: int = 6):
    """The LongHaul tool-loop protocol as an Inspect solver: one JSON action
    per turn; malformed output gets one structured retry (anomaly policy)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        tools = Tools(WORLD, state.metadata["machine_id"], state.metadata["alarms"], retriever=RETRIEVER)
        messages = [
            ChatMessageSystem(content=SYSTEM_PROMPT),
            ChatMessageUser(content=state.input_text),
        ]
        anomalies = 0
        for _ in range(max_steps):
            output = await model.generate(messages)
            text = output.completion
            obj = parse_json(text)

            if obj is None:
                anomalies += 1
                if anomalies > 1:
                    break
                messages += [
                    ChatMessageAssistant(content=text),
                    ChatMessageUser(content="Invalid output. Respond with ONE valid JSON object only."),
                ]
                continue

            if "diagnosis" in obj:
                state.store.set("diagnosis", obj["diagnosis"] or {})
                state.store.set("anomalies", anomalies)
                state.output = output
                return state

            if "tool" in obj and hasattr(tools, str(obj["tool"])):
                result = getattr(tools, obj["tool"])(**(obj.get("args") or {}))
                messages += [
                    ChatMessageAssistant(content=text),
                    ChatMessageUser(content=f"Tool result: {json.dumps(result)}"),
                ]
            else:
                anomalies += 1
                messages += [
                    ChatMessageAssistant(content=text),
                    ChatMessageUser(content="Unknown tool. Use alarm_lookup, manual_search, maintenance_history, or give a diagnosis."),
                ]

        state.store.set("diagnosis", {})
        state.store.set("anomalies", anomalies)
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def diagnosis_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        gt = json.loads(target.text)
        pred = state.store.get("diagnosis") or {}
        exact = (
            pred.get("component") == gt["component"]
            and pred.get("failure_mode") == gt["failure_mode"]
        )
        return Score(
            value=CORRECT if exact else INCORRECT,
            answer=json.dumps(pred),
            metadata={
                "component_correct": pred.get("component") == gt["component"],
                "anomalies": state.store.get("anomalies", 0),
            },
        )

    return score


@task
def longhaul() -> Task:
    return Task(dataset=episode_dataset(), solver=diagnostic_loop(), scorer=diagnosis_scorer())
