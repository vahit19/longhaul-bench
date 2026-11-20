"""Non-LLM reference points on the five incomplete-manual plants (seed 47 and g43-g46): the
manual vote rule cannot answer faults that the manual omits, whereas a learning case base can.
Same stream as the incomplete-manual runs (50 probe episodes, 945 streamed, feedback seed 7).

Usage: python scripts/nonllm_gap.py --out docs/paper/nonllm_gap.json
"""
import argparse, json, random, statistics, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from agents.memory import MemoryStore, apply_operator  # noqa: E402
from memory_replay import world_setup, SEED  # noqa: E402
from recall_precision import load_jsonl  # noqa: E402
from nonllm_baselines import manual_rule  # noqa: E402

PLANTS = {"g47": REPO / "runs/gap_world", **{f"g{s}": REPO / f"runs/gap_worlds/g{s}" for s in (43, 44, 45, 46)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=REPO / "docs/paper/nonllm_gap.json")
    args = ap.parse_args(); res = {}
    for plant, wdir in PLANTS.items():
        eps = load_jsonl(wdir / "episodes.jsonl"); setup = world_setup(wdir)
        world = json.loads((wdir / "world.json").read_text(encoding="utf-8"))
        gaps = {(m["machine_id"], c, mo) for m in world["manuals"] for c, mo in m["manual_gaps"]}
        stream = eps[50:995]; store, rng = MemoryStore(100, "compress"), random.Random(SEED)
        sym = lambda ep: [setup["alarm_symptom"][a["code"]] for a in ep["alarms"] if a["code"] in setup["alarm_symptom"]]
        hits, n_out, out_hits = Counter(), 0, Counter()
        for ep in stream:
            s, mid = sym(ep), ep["machine_id"]
            gt = (ep["ground_truth"]["component"], ep["ground_truth"]["failure_mode"])
            man = manual_rule(setup["manual_rows"].get(mid, []), s)
            cases = store.recall(mid, s)
            cbr = (cases[0]["component"], cases[0]["mode"]) if cases else man
            out = (mid, *gt) in gaps; n_out += out
            for k, v in (("manual-rule", man), ("cbr-only", cbr)):
                hits[k] += v == gt
                if out: out_hits[k] += v == gt
            apply_operator("reflect", store, mid, s, ep["ground_truth"], rng, 0.0, alternatives=setup["plausible"].get(mid))
        res[plant] = {"out_share": n_out / len(stream), **{k: v / len(stream) for k, v in hits.items()},
                      **{f"{k}_out": out_hits[k] / n_out for k in hits}}
        print(plant, {k: round(v, 3) for k, v in res[plant].items()})
    print("mean", {k: round(statistics.mean(r[k] for r in res.values()), 3) for k in next(iter(res.values()))})
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
