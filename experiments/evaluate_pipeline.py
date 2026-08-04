"""Evaluate the decision engine with diverse natural-language fault queries."""

import os
import sys
from pathlib import Path

# ── Path setup (mirrors run_diagnostic.py) ──
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "num_pipeline" / "scripts"
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(NUM_PIPELINE_DIR)

from run_diagnostic import run_diagnostic


# ── Query set ──────────────────────────────────────────────────────

QUERIES = [
    # --- Engine ---
    ("Engine overheating with coolant loss", "engine", "multi-symptom"),
    ("Check engine light is on, rough idle", "engine", "multi-symptom"),
    ("Engine misfires at idle", "engine", "exact symptom"),
    ("Car hesitates on acceleration", "engine", "paraphrase"),
    ("Loss of engine power", "engine", "exact symptom"),
    ("Engine knocking sound under load", "engine", "paraphrase"),
    ("Vehicle sputters at high speeds", "engine", "paraphrase"),
    ("Engine cranks slowly, won't start", "engine", "multi-symptom"),
    ("Engine does not crank at all", "engine", "exact symptom"),
    ("Engine stalls when I let off the gas", "engine", "conversational"),

    # --- Brakes ---
    ("Brake pedal feels spongy", "brakes", "exact symptom"),
    ("ABS warning light on, brake pedal pulsation", "brakes", "multi-symptom"),
    ("Brakes squealing when I stop", "brakes", "conversational"),
    ("Steering pulls to the left when braking", "brakes", "paraphrase"),
    ("Brake fluid leak under the car", "brakes", "exact symptom"),

    # --- Transmission ---
    ("Transmission slips between gears", "transmission", "paraphrase"),
    ("Car won't shift out of first gear", "transmission", "conversational"),
    ("Transmission grinding on upshift", "transmission", "exact symptom"),
    ("Harsh shift from 2nd to 3rd", "transmission", "vague"),

    # --- Fuel ---
    ("Poor fuel economy, running rich", "fuel", "multi-symptom"),
    ("Fuel smell after parking", "fuel", "vague"),
    ("Car won't start, no fuel pressure", "fuel", "multi-symptom"),

    # --- Cooling ---
    ("Temperature gauge goes to red", "cooling", "vague"),
    ("Coolant leaking from radiator", "cooling", "exact symptom"),
    ("Heater blows cold air, engine overheats", "cooling", "multi-symptom"),

    # --- Electrical ---
    ("Battery keeps dying overnight", "electrical", "conversational"),
    ("Dashboard lights flickering", "electrical", "vague"),
    ("Alternator warning light on", "electrical", "exact symptom"),
    ("Car electrical system not working", "electrical", "vague"),

    # --- Exhaust / Emissions ---
    ("Black smoke from exhaust", "exhaust", "exact symptom"),
    ("Check engine light, code P0301", "exhaust", "multi-symptom"),

    # --- Steering / Suspension ---
    ("Steering wheel vibrates at highway speed", "suspension", "paraphrase"),
    ("Car pulls to one side", "suspension", "vague"),
    ("Clunking noise over bumps", "suspension", "conversational"),

    # --- Vague / edge ---
    ("Something feels wrong with the car", "vague", "vague"),
    ("My car is making a weird noise", "vague", "vague"),
    ("The engine light came on yesterday", "vague", "vague"),
    ("Brakes feel weird", "vague", "partial"),
    ("Car shakes when I brake", "vague", "partial"),
]


# ── Run evaluation ─────────────────────────────────────────────────

def main():
    results = []

    for i, (query, system, qtype) in enumerate(QUERIES, 1):
        try:
            report = run_diagnostic(
                symptoms_text=query,
                current_sample="simulated",
                speed=1000,
                use_llm=False,
                verbose=False,
            )

            top_label = report.top_candidate.get("label", "N/A")
            top_fault = report.top_candidate.get("navic_fault", "N/A")
            top_score = report.top_candidate.get("confidence", 0.0)

            # Sensor evidence summary
            se = report.sensor_evidence
            sensorstatuses = {}
            for fault, ev in se.items():
                sensorstatuses[fault] = ev.get("status", "N/A")

            # Top 3 candidates
            top3 = []
            for c in report.display_candidates[:3]:
                top3.append({
                    "label": c.get("label", ""),
                    "navic_fault": c.get("navic_fault", ""),
                    "confidence": round(c.get("confidence", 0.0), 3),
                    "sensor_status": c.get("sensor_status", "N/A"),
                })

            results.append({
                "query": query,
                "system": system,
                "query_type": qtype,
                "mode": report.mode,
                "confidence": round(report.confidence, 3),
                "top_label": top_label,
                "top_fault": top_fault,
                "top_retrieval_score": round(top_score, 3),
                "sensor_statuses": sensorstatuses,
                "num_candidates": len(report.display_candidates),
                "top3": top3,
                "inspection_steps": report.inspection_steps,
            })

        except Exception as e:
            results.append({
                "query": query,
                "system": system,
                "query_type": qtype,
                "mode": "ERROR",
                "error": str(e),
            })

    # ── Print results ──────────────────────────────────────────────

    print("=" * 100)
    print("DECISION ENGINE EVALUATION RESULTS")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        print(f"\n{'-' * 100}")
        print(f"Query {i}: \"{r['query']}\"")
        print(f"  System: {r['system']}  |  Query type: {r['query_type']}")
        if r["mode"] == "ERROR":
            print(f"  *** ERROR: {r['error']} ***")
            continue
        print(f"  Mode: {r['mode']}  |  Confidence: {r['confidence']:.1%}")
        print(f"  Top fault: {r['top_label']} ({r['top_fault']})")
        print(f"  Top retrieval score: {r['top_retrieval_score']:.3f}")
        print(f"  Display candidates: {r['num_candidates']}")
        if r["sensor_statuses"]:
            print(f"  Sensor statuses: {r['sensor_statuses']}")
        else:
            print(f"  Sensor statuses: (none)")
        print(f"  Top 3 candidates:")
        for j, c in enumerate(r["top3"], 1):
            print(f"    {j}. {c['label']} ({c['navic_fault']}) "
                  f"conf={c['confidence']:.3f} sensor={c['sensor_status']}")
        if r["inspection_steps"]:
            print(f"  Inspection steps: {r['inspection_steps'][:2]}")

    # ── Summary ────────────────────────────────────────────────────

    valid = [r for r in results if r["mode"] != "ERROR"]
    errors = [r for r in results if r["mode"] == "ERROR"]

    mode_counts = {}
    for r in valid:
        mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1

    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"Total queries: {len(results)}")
    print(f"Successful:    {len(valid)}")
    print(f"Errors:        {len(errors)}")
    print()
    for mode in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
        count = mode_counts.get(mode, 0)
        pct = count / len(valid) * 100 if valid else 0
        print(f"  {mode:12s}: {count:3d}  ({pct:5.1f}%)")

    # ── Confidence stats by mode ───────────────────────────────────

    print(f"\n{'-' * 100}")
    print("CONFIDENCE DISTRIBUTION BY MODE")
    print(f"{'-' * 100}")
    for mode in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
        mode_results = [r for r in valid if r["mode"] == mode]
        if not mode_results:
            continue
        confs = [r["confidence"] for r in mode_results]
        print(f"  {mode}: min={min(confs):.3f}  max={max(confs):.3f}  "
              f"avg={sum(confs)/len(confs):.3f}  n={len(confs)}")

    # ── EXTRACTED analysis ─────────────────────────────────────────

    extracted = [r for r in valid if r["mode"] == "EXTRACTED"]
    print(f"\n{'=' * 100}")
    print(f"EXTRACTED MODE ANALYSIS ({len(extracted)} queries)")
    print(f"{'=' * 100}")

    if not extracted:
        print("  No queries classified as EXTRACTED.")
    else:
        for r in extracted:
            print(f"\n  Query: \"{r['query']}\"")
            print(f"    System: {r['system']}  |  Type: {r['query_type']}")
            print(f"    Confidence: {r['confidence']:.1%}  |  "
                  f"Top: {r['top_label']} ({r['top_fault']})")
            print(f"    Sensor: {r['sensor_statuses']}")
            print(f"    Top 3:")
            for j, c in enumerate(r["top3"], 1):
                print(f"      {j}. {c['label']} ({c['navic_fault']}) "
                      f"conf={c['confidence']:.3f} sensor={c['sensor_status']}")
            # Diagnosis
            reasons = []
            if r["confidence"] >= 0.50:
                reasons.append(f"confidence {r['confidence']:.1%} >= 0.50 threshold")
            if r["top_retrieval_score"] >= 0.30:
                reasons.append(f"retrieval score {r['top_retrieval_score']:.3f} strong")
            if any(s in ("Supported", "Contradicted") for s in r["sensor_statuses"].values()):
                reasons.append("sensor validation active")
            print(f"    Why EXTRACTED: {'; '.join(reasons) if reasons else 'mode classifier decision'}")

    # ── Sensor coverage ────────────────────────────────────────────

    print(f"\n{'-' * 100}")
    print("SENSOR VALIDATION COVERAGE")
    print(f"{'-' * 100}")
    with_sensor = sum(1 for r in valid
                      if any(s in ("Supported", "Contradicted")
                             for s in r["sensor_statuses"].values()))
    without_sensor = len(valid) - with_sensor
    print(f"  With active sensor evidence:  {with_sensor}/{len(valid)}")
    print(f"  No sensor evidence:           {without_sensor}/{len(valid)}")

    # ── System coverage ────────────────────────────────────────────

    print(f"\n{'-' * 100}")
    print("RESULTS BY VEHICLE SYSTEM")
    print(f"{'-' * 100}")
    systems = {}
    for r in valid:
        sys_name = r["system"]
        if sys_name not in systems:
            systems[sys_name] = []
        systems[sys_name].append(r)
    for sys_name, sys_results in sorted(systems.items()):
        modes = {}
        for r in sys_results:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        mode_str = ", ".join(f"{m}:{c}" for m, c in sorted(modes.items()))
        print(f"  {sys_name:15s}: {len(sys_results):2d} queries  [{mode_str}]")


if __name__ == "__main__":
    main()
