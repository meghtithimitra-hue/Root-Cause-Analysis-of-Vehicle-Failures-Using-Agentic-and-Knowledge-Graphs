"""Batch analysis of confidence components across representative queries."""
import sys, json
sys.path.insert(0, ".")
import os; os.chdir("num_pipeline")

from num_pipeline.scripts.run_diagnostic import run_diagnostic

QUERIES = [
    # High-confidence: direct symptom matches
    "engine knocking",
    "engine misfires",
    "engine stalls",
    "high fuel consumption",
    "black smoke from exhaust",
    "poor engine performance",
    "difficulty starting engine",
    # Medium: multi-word / slightly indirect
    "brake pedal feels spongy",
    "steering pulls to the left",
    "engine overheating, coolant loss",
    "ABS warning light on, brake pedal pulsation",
    "check engine light, rough idle",
    "vehicle pulls to one side during braking",
    "vibration at highway speed",
    # Low: vague or poorly matching
    "something feels wrong",
    "car making weird noise",
    "Dashboard warning light",
    "issue worsens at speed",
    "loss of engine power",
    "slow cranking",
]

results = []
for q in QUERIES:
    try:
        r = run_diagnostic(q, current_sample="simulated", speed=1000, verbose=False)
        cc = r.confidence_components
        results.append({
            "query": q,
            "mode": r.mode,
            "final": round(r.confidence, 4),
            "cal_retrieval": round(cc.get("calibrated_retrieval", 0), 4),
            "separation": round(cc.get("separation", 0), 4),
            "coverage": round(cc.get("coverage", 0), 4),
            "sensor_boost": round(cc.get("sensor_boost", 0), 4),
            "raw_top": round(cc.get("raw_retrieval_score", 0), 4),
            "top_candidate": r.top_candidate.get("label", "?")[:40],
        })
    except Exception as e:
        results.append({"query": q, "error": str(e)})

# Print as JSON
print(json.dumps(results, indent=2))
