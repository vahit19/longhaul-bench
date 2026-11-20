"""Classical case-base maintenance (CBM) baselines, compared by exact replay (no agent re-run).

All policies use the reflect write operator (a confirmation merges into an identical record and
raises its support) and the same budget of 100 records, episode streams, probe cadence and
feedback seed as the released runs (see scripts/memory_replay.py, which validated the replay
against the decision traces). Only the maintenance policy differs:

  fifo        evict the oldest record (native policy family of the released arms)
  lru         evict the least recently retrieved record
  lfu         evict the least often retrieved record
  rc          competence-preserving deletion: evict the record with the lowest relative
              coverage (Smyth & McKenna 2001), the case-deletion criterion that follows
              Smyth & Keane (1995). A record covers another if it is retrievable for it
              (same machine, at least one shared symptom) and stores the same solution.
  enn         Wilson's edited nearest neighbour rule (1972) every 50 episodes: a record is
              removed if the support-weighted majority of its k = 3 nearest other records
              (same machine, by symptom Jaccard similarity) stores a different solution;
              FIFO for the budget.
  support     support-aware eviction of the paper (released 'importance' policy:
              evict the lowest use count + 2 x support)
  enn+support ENN editing combined with support-aware eviction

The maintenance policies see only stored labels, never the ground truth. For each policy,
noise level and plant the script reports recall precision pi, the knowledge-bearing share
omega, and the paired accuracy difference against the frozen arm predicted by the
recall-precision decomposition with the conditional effects of the released reflect arm.

Usage: python scripts/cbm_baselines.py --out docs/paper/cbm_baselines.json [--latex rows.tex]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from agents.memory import MemoryStore, apply_operator  # noqa: E402
from memory_replay import world_setup, run_params, SEED  # noqa: E402
from recall_precision import WORLDS, ARMS, load_jsonl, analyse  # noqa: E402

POLICIES = ("fifo", "lru", "lfu", "rc", "enn", "support", "enn+support")
NOISE = (0.0, 0.2, 0.4)
ENN_EVERY, ENN_K = 50, 3


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


class CBMStore(MemoryStore):
    """MemoryStore with classical maintenance policies; recall and write are unchanged."""

    def __init__(self, budget_entries: int, policy: str):
        base = {"support": "importance", "enn+support": "importance"}.get(policy, "fifo")
        super().__init__(budget_entries=budget_entries, policy=base)
        self.cbm = policy
        self.clock = 0

    def recall(self, machine_id, symptoms, k=3):
        top = super().recall(machine_id, symptoms, k)
        self.clock += 1
        for e in top:
            e["last"] = self.clock
        return top

    def _relative_coverage(self) -> list:
        ents = self.entries
        reach = [0] * len(ents)
        covers = [[] for _ in ents]
        for i, a in enumerate(ents):
            sa = set(a["symptoms"])
            for j, b in enumerate(ents):
                if (a["machine_id"] == b["machine_id"] and (a["component"], a["mode"]) ==
                        (b["component"], b["mode"]) and sa & set(b["symptoms"])):
                    covers[i].append(j)
                    reach[j] += 1
        return [sum(1 / reach[j] for j in covers[i]) for i in range(len(ents))]

    def _enforce_budget(self) -> None:
        if self.cbm in ("fifo", "enn", "support", "enn+support"):
            return super()._enforce_budget()
        # cache semantics: insertion counts as a use for LRU, and LFU evicts before inserting,
        # so a record that has just been written is never the LRU/LFU victim
        newest = None
        for e in self.entries:
            if "t_in" not in e:
                e["t_in"] = self.clock
                newest = e
        while len(self.entries) > self.budget:
            if self.cbm == "lru":
                victim = min(self.entries, key=lambda e: max(e.get("last", 0), e["t_in"]) + (e is newest) * 1e9)
            elif self.cbm == "lfu":
                victim = min((e for e in self.entries if e is not newest), key=lambda e: e["uses"])
            elif self.cbm == "rc":
                rc = self._relative_coverage()
                victim = self.entries[min(range(len(rc)), key=lambda i: rc[i])]
            else:
                raise ValueError(self.cbm)
            self.entries.remove(victim)
            self.evictions += 1

    def enn_edit(self) -> int:
        """Wilson's ENN over the records; support counts act as instance multiplicities."""
        drop = []
        for i, a in enumerate(self.entries):
            if a["count"] >= ENN_K:  # its own duplicates form the k nearest neighbours
                continue
            scored = [(jaccard(a["symptoms"], b["symptoms"]), j, b) for j, b in enumerate(self.entries)
                      if j != i and b["machine_id"] == a["machine_id"]]
            # most similar first; ties broken by higher support, then by insertion order
            scored.sort(key=lambda x: (-x[0], -x[2]["count"], x[1]))
            nbrs = [b for sim, _, b in scored if sim > 0]
            votes, need = {}, ENN_K - (a["count"] - 1)
            same = a["count"] - 1
            for b in nbrs:
                if need <= 0:
                    break
                w = min(b["count"], need)
                votes[(b["component"], b["mode"])] = votes.get((b["component"], b["mode"]), 0) + w
                need -= w
            if need > 0:  # fewer than k neighbours: keep
                continue
            agree = same + votes.get((a["component"], a["mode"]), 0)
            if agree * 2 < ENN_K:
                drop.append(i)
        for i in reversed(drop):
            self.entries.pop(i)
        return len(drop)


def replay(setup, all_eps, stream_ids, probe_size, probe_every, policy, noise, budget=100, return_labels=False):
    sym = lambda ep: [setup["alarm_symptom"][a["code"]] for a in ep["alarms"]
                      if a["code"] in setup["alarm_symptom"]]
    by_id = {e["episode_id"]: e for e in all_eps}
    probes = all_eps[:probe_size]
    store = CBMStore(budget, policy)
    rng = random.Random(SEED)
    labels, edited = {}, 0
    for i, eid in enumerate(stream_ids, 1):
        ep = by_id[eid]
        s = sym(ep)
        cases = store.recall(ep["machine_id"], s)
        gt = (ep["ground_truth"]["component"], ep["ground_truth"]["failure_mode"])
        labels[eid] = ("none" if not cases else
                       "correct" if any((c["component"], c["mode"]) == gt for c in cases) else "misleading")
        apply_operator("reflect", store, ep["machine_id"], s, ep["ground_truth"], rng, noise,
                       alternatives=setup["plausible"].get(ep["machine_id"]))
        if policy.startswith("enn") and i % ENN_EVERY == 0:
            edited += store.enn_edit()
        if i % probe_every == 0 or i == len(stream_ids):
            for pe in probes:
                store.recall(pe["machine_id"], sym(pe))
    n = len(labels)
    c = sum(v == "correct" for v in labels.values())
    w = sum(v == "misleading" for v in labels.values())
    out = {"pi": c / (c + w) if c + w else None, "omega": (c + w) / n, "edited": edited}
    if return_labels:
        out["labels"] = labels
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "docs" / "paper" / "cbm_baselines.json")
    p.add_argument("--latex", type=Path, default=None)
    args = p.parse_args()

    res, effects = {}, {}
    for world, ep_path in WORLDS.items():
        all_eps = load_jsonl(ep_path)
        setup = world_setup(ep_path.parent)
        stream_ids = [r["episode_id"] for r in load_jsonl(REPO / "runs" / world / "frozen" / "results.jsonl")]
        probe_size, probe_every = run_params(world, all_eps, stream_ids)
        a = analyse(world, ARMS["reflect"][world])
        e = effects[world] = {k: a[k] for k in ("d_plus", "d_minus", "d_none")}
        for pol in POLICIES:
            for noise in NOISE:
                st = replay(setup, all_eps, stream_ids, probe_size, probe_every, pol, noise)
                d0 = e.get("d_none") or 0.0
                st["pred_diff"] = st["omega"] * (st["pi"] * e["d_plus"] + (1 - st["pi"]) * e["d_minus"]) \
                    + (1 - st["omega"]) * d0
                res.setdefault(world, {})[f"{pol}|{noise}"] = st
                print(f"{world} {pol:12s} p={noise} pi={st['pi']:.3f} omega={st['omega']:.3f} "
                      f"pred={100 * st['pred_diff']:+.1f} edited={st['edited']}", flush=True)

    mean = {}
    for pol in POLICIES:
        for noise in NOISE:
            cells = [res[w][f"{pol}|{noise}"] for w in res]
            ref = [res[w][f"fifo|{noise}"]["pi"] for w in res]
            sup = [res[w][f"support|{noise}"]["pi"] for w in res]
            mean[f"{pol}|{noise}"] = {
                "pi": statistics.mean(c["pi"] for c in cells),
                "pi_min": min(c["pi"] for c in cells), "pi_max": max(c["pi"] for c in cells),
                "omega": statistics.mean(c["omega"] for c in cells),
                "pred_diff": statistics.mean(c["pred_diff"] for c in cells),
                "plants_above_fifo": sum(c["pi"] > r for c, r in zip(cells, ref)),
                "plants_above_support": sum(c["pi"] > r for c, r in zip(cells, sup)),
            }
    out = {"settings": {"budget": 100, "write_operator": "reflect", "enn_every": ENN_EVERY, "enn_k": ENN_K,
                        "seed": SEED}, "effects": effects, "per_plant": res, "mean": mean}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    for k, v in mean.items():
        print(f"{k:18s} pi={v['pi']:.3f} [{v['pi_min']:.3f},{v['pi_max']:.3f}] omega={v['omega']:.3f} "
              f"pred={100 * v['pred_diff']:+.1f} >fifo:{v['plants_above_fifo']}/5 >support:{v['plants_above_support']}/5")
    if args.latex:
        names = {"fifo": "FIFO", "lru": "LRU", "lfu": "LFU", "rc": "Relative coverage",
                 "enn": "ENN editing", "support": "Support-aware", "enn+support": "ENN + support-aware"}
        rows = [f"{names[pol]} & " + " & ".join(f"{mean[f'{pol}|{n}']['pi']:.3f}" for n in NOISE)
                + f" & ${100 * mean[f'{pol}|0.4']['pred_diff']:+.1f}$ & {mean[f'{pol}|0.4']['plants_above_fifo']}/5 \\\\"
                for pol in POLICIES]
        args.latex.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("wrote", args.latex)


if __name__ == "__main__":
    main()
