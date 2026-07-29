"""Iteratively discover queries that produce INFERRED mode."""

import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "num_pipeline" / "scripts"
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(NUM_PIPELINE_DIR)

# Suppress evidence_fusion noise
import pipeline.evidence_fusion as ef
_orig = ef.fuse_evidence
def _quiet(retrieval_result, mapped_faults, sensor_results):
    with contextlib.redirect_stdout(io.StringIO()):
        return _orig(retrieval_result, mapped_faults, sensor_results)
ef.fuse_evidence = _quiet

import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from run_diagnostic import run_diagnostic

EXTRACTED_THRESHOLD = 0.75
INFERRED_LOW = 0.40

candidates_tested = 0
inferred_found = 0

def test_queries(queries, label, found_list):
    global candidates_tested, inferred_found
    print(f"\n{'='*100}")
    print(f"ROUND: {label} ({len(queries)} queries)")
    print(f"{'='*100}")
    round_inferred = []
    for q in queries:
        candidates_tested += 1
        try:
            report = run_diagnostic(
                symptoms_text=q,
                current_sample="simulated",
                speed=1000,
                use_llm=False,
                verbose=False,
            )
        except Exception as e:
            print(f"  ERROR [{q}]: {e}")
            continue

        mode = report.mode
        conf = report.confidence
        top = report.top_candidate.get("label", "N/A")
        fault = report.top_candidate.get("navic_fault", "N/A")

        if mode == "INFERRED":
            inferred_found += 1
            round_inferred.append((q, round(conf, 4), top, fault))
            print(f"  INFERRED [{q:55s}] conf={conf:.4f}  top=\"{top}\" ({fault})")
            found_list.append((q, conf, top, fault))
        else:
            print(f"  {mode:10s} [{q:55s}] conf={conf:.4f}  top=\"{top}\" ({fault})")
    return round_inferred

found = []

# ── Round 1: Systems that had INFERRED before, plus variants ──
# Key insight: INFERRED happens when:
#  - cal is moderate (<1.0) OR sep+cov are low enough to not breach 0.75
#  - Tied candidates (low sep) is a reliable pattern
#  - Sensor-unmapped faults with moderate retrieval

round1 = [
    # Brakes: tied candidates should reduce separation
    "Brake pedal feels soft",
    "Brake pedal goes to floor",
    "Brake warning light stays on",
    # Engine: paraphrases that partially match
    "Engine is running rough after start",
    "Engine vibrates at idle",
    "Car idles roughly when warm",
    # Cooling: moderate KG coverage
    "Radiator fan not spinning",
    "Engine running hot",
    "Low coolant warning light",
    # Electrical: poor KG coverage = lower retrieval
    "Radio not working",
    "Horn not working",
    "Power windows not working",
    # Transmission: vague terms
    "Gearbox making noise",
    "Hard to get into gear",
    "Transmission fluid leaking",
]

test_queries(round1, "1 - Broad system coverage", found)

# ── Round 2: Target low-separation (tied candidates) ──
# Tied candidates → sep ≈ 0 → removes 0.20 contribution
# Need raw retrieval moderate + low cov to stay under 0.75

print(f"\n{'─'*80}")
print(f"After Round 1: {len(found)} INFERRED found ({candidates_tested} tested)")

round2 = [
    # Queries with generic terms that tie multiple KG entities
    "Warning light is on",
    "Car not running properly",
    "Strange vibration from front",
    "Fluid leak underneath car",
    "Engine warning light flashing",
    "Car is shaking badly",
    "Burning smell from engine bay",
    "Oil pressure light flickers",
    "Tire pressure warning light",
    "Battery light on dashboard",
]

test_queries(round2, "2 - Generic / tied-candidate patterns", found)

print(f"\nAfter Round 2: {len(found)} INFERRED found ({candidates_tested} tested)")

# ── Round 3: Focus on systems with partial KG coverage ──
# Cooling, exhaust, suspension, electrical are poorly covered
# Use exact phrases that partially overlap existing entities

round3 = [
    # Exhaust variants
    "Smoke from tailpipe",
    "Exhaust smells like gas",
    "Loud exhaust noise",
    "Exhaust rattling sound",
    # Suspension variants  
    "Steering feels loose",
    "Car drifting to right",
    "Bumpy ride over small bumps",
    "Suspension creaking noise",
    "Wheel shaking at low speed",
    "Steering wheel off center",
    # Cooling variants
    "Coolant smell inside car",
    "Engine temperature high",
    "Water pump noise",
    "Thermostat stuck closed",
    # Electrical variants
    "Interior lights flickering",
    "Headlights dim while driving",
    "Electrical burning smell",
    "Accessory power not working",
    "Starter motor clicking",
    # Fuel system variants
    "Smell of gas inside car",
    "Engine surges at cruise",
    "Fuel gauge not accurate",
    "Hard start when cold",
]

test_queries(round3, "3 - Partial-coverage systems", found)

print(f"\nAfter Round 3: {len(found)} INFERRED found ({candidates_tested} tested)")

# ── Round 4: if we still don't have 4, try very specific strategies ──
# Strategy: queries with known KG entities that tie in score
# (same retrieval score from multiple candidates = low sep)
# Strategy: queries that match entity labels well but don't cover all words

if len(found) < 4:
    round4 = [
        # Specific queries targeting entity label matches with partial coverage
        "Engine light is on but car runs fine",
        "Squeaking when turning steering wheel",
        "Knocking from front when going over bumps",
        "Whining noise from transmission area",
        "Car jerks when accelerating uphill",
        "Vibration in steering wheel at certain speeds",
        "Clicking sound when turning left",
        "Engine backfires on deceleration",
        "High-pitched squeal from belts",
        "Metallic grinding when braking slowly",
        "Check engine light blinking under load",
        "Engine oil leak near filter",
        "Coolant temperature warning light blinking",
        "Battery terminals corroded",
        "Strong gasoline odor from oil dipstick",
        "White smoke from exhaust when cold start",
        "Car accelerates slowly in hot weather",
        "Rattling from exhaust on cold start",
        "Brake pedal vibrates under hard braking",
        "Transmission hunts between gears on incline",
    ]
    test_queries(round4, "4 - Specific compound symptoms", found)

# ── Round 5: if still not enough, try every reasonable variation ──

if len(found) < 4:
    round5 = [
        "Idle speed too high",
        "Engine stalls at stop signs",
        "Blinking check engine light",
        "Spark plug misfire codes",
        "Car battery dies after sitting",
        "Door locks not working",
        "Cruise control not engaging",
        "Seat heater not working",
        "Rear defroster not working",
        "Windshield washer not spraying",
        "Turn signal relay clicking fast",
        "Hazard lights not flashing",
        "Brake lights stuck on",
        "Fog lights not working",
        "Dashboard dim at night",
        "Air conditioning blows warm air",
        "Heater only works on high",
        "Cooling fan runs constantly",
        "Engine ticks when cold",
        "Exhaust pipe rusted through",
    ]
    test_queries(round5, "5 - Auxiliary system faults", found)

# ── Final report ──

print(f"\n{'='*100}")
print(f"RESULTS: {len(found)} INFERRED queries found out of {candidates_tested} tested")
print(f"{'='*100}")

# Deduplicate by exact query text
seen = set()
unique = []
for q, conf, top, fault in found:
    if q not in seen:
        seen.add(q)
        unique.append((q, conf, top, fault))

print(f"\nAll unique INFERRED queries ({len(unique)} total):")
for i, (q, conf, top, fault) in enumerate(unique, 1):
    print(f"  {i:2d}. \"{q}\"")
    print(f"      conf={conf:.4f}  top=\"{top}\"  fault={fault}")

print(f"\nFirst {min(4, len(unique))} INFERRED queries as a programmatic list:")
print("inferred_set = [")
for q, conf, top, fault in unique[:4]:
    print(f"    \"{q}\",")
print("]")
