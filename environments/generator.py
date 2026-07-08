"""Synthetic industrial world generator for LongHaul-Bench.

Generates a reproducible mini-factory: machines with components, an alarm-code
table, manual excerpts (the RAG corpus), maintenance history, and a stream of
diagnostic episodes whose hidden root cause is known — so an agent's diagnosis
can be scored objectively.

The entire world is determined by --seed: same seed, same factory, same
episodes, on any machine.

Usage:
    python environments/generator.py --machines 5 --episodes 50 --seed 42 --out runs/demo
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Domain model: machine types, components, failure modes
# --------------------------------------------------------------------------

MACHINE_TYPES: dict[str, list[str]] = {
    "hydraulic_pump": ["electric_motor", "bearing", "hydraulic_valve", "pressure_sensor", "filter"],
    "air_compressor": ["electric_motor", "bearing", "belt", "temperature_sensor", "filter"],
    "conveyor": ["electric_motor", "gearbox", "belt", "speed_sensor"],
    "cnc_mill": ["spindle", "bearing", "coolant_pump", "temperature_sensor"],
    "packaging_robot": ["servo_motor", "gearbox", "proximity_sensor", "pneumatic_valve"],
}

# component -> failure mode -> (observable symptoms, log template, repair action)
FAILURE_MODES: dict[str, dict[str, dict] ] = {
    "electric_motor": {
        "winding_fault": {
            "symptoms": ["overcurrent", "temperature_high"],
            "log": "phase current imbalance {a:.1f}A / {b:.1f}A / {c:.1f}A",
            "repair": "rewind or replace motor",
        },
        "misalignment": {
            "symptoms": ["vibration_high"],
            "log": "vibration RMS {v:.2f} mm/s, dominant at 2x running speed",
            "repair": "realign motor-load coupling",
        },
    },
    "servo_motor": {
        "encoder_fault": {
            "symptoms": ["position_error"],
            "log": "following error {e:.2f} mm exceeds window",
            "repair": "replace encoder",
        },
    },
    "bearing": {
        "wear": {
            "symptoms": ["vibration_high", "temperature_high"],
            "log": "envelope spectrum shows BPFO harmonics, temp {t:.1f} C",
            "repair": "replace bearing",
        },
        "lubrication_loss": {
            "symptoms": ["temperature_high", "noise"],
            "log": "bearing temp {t:.1f} C rising, audible grinding reported",
            "repair": "relubricate and inspect seals",
        },
    },
    "hydraulic_valve": {
        "stiction": {
            "symptoms": ["pressure_oscillation"],
            "log": "pressure oscillating {lo:.1f}-{hi:.1f} bar at 0.5 Hz",
            "repair": "clean or replace valve spool",
        },
    },
    "pneumatic_valve": {
        "leak": {
            "symptoms": ["pressure_drop", "cycle_slow"],
            "log": "actuation time {t:.2f}s vs nominal 0.40s, supply pressure sagging",
            "repair": "replace valve seals",
        },
    },
    "pressure_sensor": {
        "drift": {
            "symptoms": ["reading_implausible"],
            "log": "static pressure reads {p:.1f} bar while pump is off",
            "repair": "recalibrate or replace sensor",
        },
    },
    "temperature_sensor": {
        "drift": {
            "symptoms": ["reading_implausible"],
            "log": "ambient reported as {t:.1f} C, plant ambient is 24 C",
            "repair": "recalibrate or replace sensor",
        },
    },
    "speed_sensor": {
        "intermittent": {
            "symptoms": ["signal_dropout"],
            "log": "speed signal dropouts {n} times in last hour",
            "repair": "reseat connector, replace cable if worn",
        },
    },
    "proximity_sensor": {
        "misadjustment": {
            "symptoms": ["missed_detection"],
            "log": "{n} missed part detections in last shift",
            "repair": "readjust sensing distance",
        },
    },
    "filter": {
        "clogging": {
            "symptoms": ["flow_low", "temperature_high", "pressure_drop"],
            "log": "differential pressure {dp:.2f} bar across filter (limit 0.8)",
            "repair": "replace filter element",
        },
    },
    "belt": {
        "slippage": {
            "symptoms": ["speed_unstable", "noise"],
            "log": "output speed variance {v:.1f}% under load, squeal reported",
            "repair": "retension or replace belt",
        },
    },
    "gearbox": {
        "tooth_wear": {
            "symptoms": ["vibration_high", "noise"],
            "log": "gear mesh frequency sidebands rising, oil sample shows metal particles",
            "repair": "inspect gears, replace worn stage",
        },
    },
    "spindle": {
        "imbalance": {
            "symptoms": ["vibration_high", "surface_finish_poor"],
            "log": "vibration {v:.2f} mm/s at 1x spindle speed, part finish out of spec",
            "repair": "balance spindle, check tool holder",
        },
    },
    "coolant_pump": {
        "cavitation": {
            "symptoms": ["flow_low", "noise"],
            "log": "coolant flow {f:.1f} L/min vs nominal 20, rattling noise",
            "repair": "clear suction line, check coolant level",
        },
    },
}

# Symptom -> alarm message text. Note deliberate ambiguity: several components
# can raise the same symptom (e.g. temperature_high), so a correct diagnosis
# requires consulting manuals/logs, not just the alarm table.
SYMPTOM_ALARMS: dict[str, str] = {
    "overcurrent": "Motor overcurrent protection triggered",
    "temperature_high": "Temperature above warning threshold",
    "vibration_high": "Vibration level above ISO 10816 zone B",
    "pressure_oscillation": "Hydraulic pressure unstable",
    "pressure_drop": "Supply pressure below nominal",
    "reading_implausible": "Sensor plausibility check failed",
    "signal_dropout": "Sensor signal intermittent",
    "missed_detection": "Part detection failure",
    "flow_low": "Flow rate below nominal",
    "speed_unstable": "Output speed unstable",
    "cycle_slow": "Cycle time above nominal",
    "position_error": "Servo following error",
    "noise": "Abnormal noise reported by operator",
    "surface_finish_poor": "Quality check: surface finish out of spec",
}


@dataclass
class Machine:
    machine_id: str
    machine_type: str
    components: list[str]
    installed_year: int


@dataclass
class Episode:
    episode_id: str
    machine_id: str
    alarms: list[dict]
    log_excerpt: list[str]
    operator_note: str
    # hidden from the agent at inference time; used only for scoring
    ground_truth: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def build_machines(rng: random.Random, n: int) -> list[Machine]:
    types = list(MACHINE_TYPES)
    return [
        Machine(
            machine_id=f"M{i+1:02d}",
            machine_type=(t := types[i % len(types)] if i < len(types) else rng.choice(types)),
            components=list(MACHINE_TYPES[t]),
            installed_year=rng.randint(2008, 2024),
        )
        for i in range(n)
    ]


def build_alarm_table(machines: list[Machine]) -> list[dict]:
    table, code = [], 100
    for m in machines:
        symptoms = sorted({s for c in m.components for s in _component_symptoms(c)})
        for s in symptoms:
            table.append({
                "code": f"{m.machine_id}-A{code}",
                "machine_id": m.machine_id,
                "symptom": s,
                "message": SYMPTOM_ALARMS[s],
            })
            code += 1
    return table


def _component_symptoms(component: str) -> set[str]:
    return {s for mode in FAILURE_MODES.get(component, {}).values() for s in mode["symptoms"]}


def build_manual(m: Machine) -> dict:
    """Manual excerpt per machine: the RAG corpus. Troubleshooting rows map a
    symptom to *all* components that can cause it — the agent must disambiguate."""
    cause_map: dict[str, list[str]] = {}
    for c in m.components:
        for mode_name, mode in FAILURE_MODES.get(c, {}).items():
            for s in mode["symptoms"]:
                cause_map.setdefault(s, []).append(f"{c} ({mode_name})")
    return {
        "machine_id": m.machine_id,
        "title": f"{m.machine_type.replace('_', ' ').title()} — Service Manual (excerpt)",
        "sections": [
            {
                "heading": "Troubleshooting",
                "rows": [
                    {"symptom": s, "possible_causes": causes, "note": "Confirm with logs before replacement."}
                    for s, causes in sorted(cause_map.items())
                ],
            }
        ],
    }


def build_maintenance_history(rng: random.Random, m: Machine, entries: int = 4) -> list[dict]:
    hist = []
    for _ in range(entries):
        c = rng.choice(m.components)
        modes = list(FAILURE_MODES.get(c, {}))
        if not modes:
            continue
        mode = rng.choice(modes)
        hist.append({
            "machine_id": m.machine_id,
            "date": f"202{rng.randint(3, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "component": c,
            "action": FAILURE_MODES[c][mode]["repair"],
        })
    return sorted(hist, key=lambda h: h["date"])


def _fill_log(template: str, rng: random.Random) -> str:
    values = {
        "a": rng.uniform(8, 14), "b": rng.uniform(8, 14), "c": rng.uniform(15, 22),
        "v": rng.uniform(4.5, 11.0), "t": rng.uniform(78, 105),
        "lo": rng.uniform(80, 95), "hi": rng.uniform(110, 130),
        "p": rng.uniform(3, 9), "dp": rng.uniform(0.9, 1.6),
        "f": rng.uniform(6, 12), "e": rng.uniform(0.4, 1.2),
        "n": rng.randint(3, 17),
    }
    return template.format(**values)


def build_episode(rng: random.Random, i: int, machines: list[Machine], alarm_table: list[dict],
                  log_dropout: float = 0.3, symptom_dropout: float = 0.3) -> Episode:
    m = rng.choice(machines)
    component = rng.choice([c for c in m.components if FAILURE_MODES.get(c)])
    mode_name = rng.choice(list(FAILURE_MODES[component]))
    mode = FAILURE_MODES[component][mode_name]

    lookup = {(a["machine_id"], a["symptom"]): a for a in alarm_table}
    alarms = [lookup[(m.machine_id, s)] for s in mode["symptoms"] if (m.machine_id, s) in lookup]
    # difficulty knob: sensors miss events — each true symptom alarm fires
    # with probability (1 - symptom_dropout); at least one always fires
    kept = [a for a in alarms if rng.random() >= symptom_dropout]
    alarms = kept if kept else ([rng.choice(alarms)] if alarms else [])
    # noise: 0-2 spurious, unrelated alarms fire too
    n_noise = rng.choices([0, 1, 2], weights=[0.45, 0.4, 0.15])[0]
    others = [a for a in alarm_table if a["machine_id"] == m.machine_id and a not in alarms]
    for a in rng.sample(others, min(n_noise, len(others))):
        alarms.append(a)
    rng.shuffle(alarms)

    # difficulty knob: with probability `log_dropout` the detailed log is
    # unavailable and the agent must reason from ambiguous symptoms alone
    if rng.random() < log_dropout:
        log_excerpt = [f"[{m.machine_id}] no detailed log available for this time window"]
    else:
        log_excerpt = [f"[{m.machine_id}] " + _fill_log(mode["log"], rng)]

    return Episode(
        episode_id=f"E{i+1:05d}",
        machine_id=m.machine_id,
        alarms=[{"code": a["code"], "message": a["message"]} for a in alarms],
        log_excerpt=log_excerpt,
        operator_note=rng.choice([
            "Machine behaving oddly since start of shift.",
            "Operator requests inspection before next batch.",
            "Issue appeared gradually over the last two days.",
            "Problem occurs mainly under full load.",
        ]),
        ground_truth={
            "component": component,
            "failure_mode": mode_name,
            "repair": mode["repair"],
        },
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate a seeded synthetic industrial world.")
    p.add_argument("--machines", type=int, default=5)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dropout", type=float, default=0.3,
                   help="probability that an episode has no detailed log (difficulty knob)")
    p.add_argument("--symptom-dropout", type=float, default=0.3,
                   help="probability that each true symptom alarm fails to fire (difficulty knob)")
    p.add_argument("--out", type=Path, default=Path("runs/demo"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    machines = build_machines(rng, args.machines)
    alarm_table = build_alarm_table(machines)
    world = {
        "seed": args.seed,
        "machines": [asdict(m) for m in machines],
        "alarm_table": alarm_table,
        "manuals": [build_manual(m) for m in machines],
        "maintenance_history": [h for m in machines for h in build_maintenance_history(rng, m)],
    }
    episodes = [build_episode(rng, i, machines, alarm_table, args.log_dropout, args.symptom_dropout) for i in range(args.episodes)]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "world.json").write_text(json.dumps(world, indent=2), encoding="utf-8")
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(asdict(e)) + "\n")

    print(f"world:    {args.out / 'world.json'}  ({len(machines)} machines, {len(alarm_table)} alarm codes)")
    print(f"episodes: {args.out / 'episodes.jsonl'}  ({len(episodes)} episodes)")
    print(f"seed:     {args.seed} (rerun with same seed -> identical world)")


if __name__ == "__main__":
    main()
