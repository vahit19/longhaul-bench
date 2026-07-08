"""RAG-quality metrics for LongHaul-Bench retrieval, via DeepEval.

Evaluates the `manual_search` retrieval step with DeepEval's contextual
relevancy metric, using the LOCAL quantized SLM as the judge (no cloud).
Limitation stated for the paper: a 3B judge is demo-grade; the full analysis
will replicate with a larger judge and human spot-checks.

Usage (both llama-servers running: 8080 chat, 8081 embeddings):
    local/venv/Scripts/python scripts/rag_metrics.py --limit 5 --retrieval vector
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate  # noqa: E402
from deepeval.metrics import ContextualRelevancyMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from openai import OpenAI  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agents.slm_agent import Tools  # noqa: E402


class LocalJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the local llama-server (OpenAI-compatible)."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8080/v1"):
        self.client = OpenAI(base_url=endpoint, api_key="sk-local")

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        text = self.client.chat.completions.create(
            model="qwen2.5-3b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        ).choices[0].message.content
        if schema is not None:  # DeepEval passes a pydantic schema for JSON outputs
            from json import loads
            start = text.find("{")
            return schema.model_validate(loads(text[start:text.rfind("}") + 1]))
        return text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return "qwen2.5-3b-instruct-local"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--retrieval", choices=["keyword", "vector"], default="vector")
    args = p.parse_args()

    world = json.loads((REPO / "runs/v01/world.json").read_text(encoding="utf-8"))
    episodes = [json.loads(l) for l in (REPO / "runs/v01/episodes.jsonl").open(encoding="utf-8")][: args.limit]

    retriever = None
    if args.retrieval == "vector":
        from environments.retrieval import VectorIndex
        retriever = VectorIndex(world)

    alarm_symptom = {a["code"]: a["symptom"] for a in world["alarm_table"]}
    cases = []
    for ep in episodes:
        symptoms = [alarm_symptom.get(a["code"], "") for a in ep["alarms"]]
        query = "diagnose fault with symptoms: " + ", ".join(s for s in symptoms if s)
        tools = Tools(world, ep["machine_id"], ep["alarms"], retriever=retriever)
        rows = tools.manual_search(query=query)
        cases.append(LLMTestCase(
            input=query,
            actual_output=json.dumps(ep["ground_truth"]),
            retrieval_context=[json.dumps(r) for r in rows],
        ))

    metric = ContextualRelevancyMetric(threshold=0.5, model=LocalJudge(), include_reason=False, async_mode=False)
    result = evaluate(test_cases=cases, metrics=[metric])
    scores = [tr.metrics_data[0].score for tr in result.test_results if tr.metrics_data and tr.metrics_data[0].score is not None]
    print(json.dumps({
        "retrieval_mode": args.retrieval,
        "cases": len(cases),
        "contextual_relevancy_mean": round(sum(scores) / len(scores), 3) if scores else None,
        "judge": "qwen2.5-3b-instruct-local (demo-grade; larger judge planned for full analysis)",
    }, indent=2))


if __name__ == "__main__":
    main()
