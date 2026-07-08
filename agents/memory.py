"""Knowledge state (memory) and improvement operators for LongHaul-Bench.

The agent accumulates experience across episodes. After each episode the
environment provides outcome feedback (modeling a technician confirming the
repair — optionally noisy). An improvement operator decides what enters the
knowledge state; a forgetting policy decides what leaves it when the memory
budget is exceeded. The central question of the benchmark lives here:
which (operator, policy) pairs improve reliability over 1000+ episodes,
and which corrupt it?

Operators:
    frozen   — no learning (control)
    append   — store every confirmed case verbatim
    reflect  — distill cases into merged rules (machine, symptom-set) -> diagnosis
    gated    — error-driven: store only cases the current memory gets wrong

Forgetting policies (applied when over budget):
    fifo        — evict oldest
    importance  — evict lowest score (uses + confirmations)
    compress    — merge duplicates first, then fifo
"""

from __future__ import annotations

import json
import random


class MemoryStore:
    def __init__(self, budget_entries: int = 100, policy: str = "fifo"):
        self.budget = budget_entries
        self.policy = policy
        self.entries: list = []  # {machine_id, symptoms, component, mode, count, uses, confirmed}
        self.evictions = 0

    # ---- recall -----------------------------------------------------------
    def recall(self, machine_id: str, symptoms: list, k: int = 3) -> list:
        scored = []
        sset = set(symptoms)
        for e in self.entries:
            if e["machine_id"] != machine_id:
                continue
            overlap = len(sset & set(e["symptoms"]))
            if overlap:
                scored.append((overlap, e))
        scored.sort(key=lambda x: (-x[0], -x[1]["count"]))
        top = [e for _, e in scored[:k]]
        for e in top:
            e["uses"] += 1
        return top

    def predict(self, machine_id: str, symptoms: list):
        """Memory's own best guess (used by the gated operator)."""
        top = self.recall(machine_id, symptoms, k=1)
        return (top[0]["component"], top[0]["mode"]) if top else None

    def render(self, cases: list) -> str:
        lines = [
            f"- symptoms {', '.join(e['symptoms'])} -> {e['component']} ({e['mode']}), "
            f"seen {e['count']}x"
            for e in cases
        ]
        return "Past confirmed cases for this machine:\n" + "\n".join(lines)

    # ---- write ------------------------------------------------------------
    def add(self, machine_id: str, symptoms: list, component: str, mode: str, merge: bool = False) -> None:
        if merge:
            for e in self.entries:
                if (e["machine_id"] == machine_id and set(e["symptoms"]) == set(symptoms)
                        and e["component"] == component and e["mode"] == mode):
                    e["count"] += 1
                    return
        self.entries.append({
            "machine_id": machine_id, "symptoms": sorted(symptoms),
            "component": component, "mode": mode, "count": 1, "uses": 0,
        })
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        if self.policy == "compress":
            merged: dict = {}
            for e in self.entries:
                key = (e["machine_id"], tuple(e["symptoms"]), e["component"], e["mode"])
                if key in merged:
                    merged[key]["count"] += e["count"]
                    merged[key]["uses"] += e["uses"]
                else:
                    merged[key] = e
            self.entries = list(merged.values())
        while len(self.entries) > self.budget:
            if self.policy == "importance":
                victim = min(self.entries, key=lambda e: e["uses"] + 2 * e["count"])
                self.entries.remove(victim)
            else:  # fifo (also compress fallback)
                self.entries.pop(0)
            self.evictions += 1

    # ---- accounting -------------------------------------------------------
    def size_bytes(self) -> int:
        return len(json.dumps(self.entries).encode())

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes": self.size_bytes(), "evictions": self.evictions}


def apply_operator(operator: str, store: MemoryStore, machine_id: str, symptoms: list,
                   gt: dict, rng: random.Random, feedback_noise: float = 0.0) -> None:
    """Update the knowledge state after an episode, per the chosen operator.

    Feedback models post-repair confirmation; with probability feedback_noise
    the recorded outcome is corrupted (wrong component from the same machine's
    plausible set) — the lever for corruption dose-response experiments.
    """
    if operator == "frozen":
        return

    component, mode = gt["component"], gt["failure_mode"]
    if feedback_noise and rng.random() < feedback_noise:
        component = component + "_misdiagnosed"  # deliberately corrupt record

    if operator == "append":
        store.add(machine_id, symptoms, component, mode, merge=False)
    elif operator == "reflect":
        store.add(machine_id, symptoms, component, mode, merge=True)
    elif operator == "gated":
        if store.predict(machine_id, symptoms) != (component, mode):
            store.add(machine_id, symptoms, component, mode, merge=True)
    else:
        raise ValueError(f"unknown operator: {operator}")
