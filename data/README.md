# LongHaul-Bench canonical datasets (v0)

These are the OFFICIAL evaluation sets. All reported numbers must reference one
of these releases by name; ad-hoc regenerations belong in `runs/`, not here.

| Release | Episodes | Difficulty knobs | Generator seed |
|---|---|---|---|
| `longhaul-v0-standard` | 1000 | log-dropout 0.3, symptom-dropout 0.3 | 42 |
| `longhaul-v0-hard` | 1000 | log-dropout 0.7, symptom-dropout 0.5 | 42 |

Each release contains:
- `world.json` — machines, alarm-code table, manual excerpts (RAG corpus), maintenance history
- `episodes.jsonl` — diagnostic episodes; `ground_truth` is for scoring only and must be hidden from the agent

Regeneration is byte-identical:

```bash
python environments/generator.py --machines 5 --episodes 1000 --seed 42 \
    --log-dropout 0.3 --symptom-dropout 0.3 --out data/longhaul-v0-standard
```

Versioning policy: any change to the generator that alters outputs bumps the
release name (v0 → v1); old releases stay for comparability. A Hugging Face
Datasets mirror is planned alongside the arXiv preprint.

Provenance note: the v0 releases are FROZEN artifacts produced by the generator
revision current at release time. A later generator refactor (adding the
`--calibrated` fault-frequency option) changed random-stream consumption, so
regenerating v0 byte-identically requires checking out the pre-refactor
revision; with the current generator, per-seed determinism still holds (CI-tested)
but yields a different stream. Integrity of the released files is verified in CI
against `data/SHA256SUMS`.
