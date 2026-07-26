"""
run_diagnostic.py — Pipeline wrapper for the new decision engine.

Runs the existing pipeline steps 1–5 (preprocessing, retrieval,
mapping, sensor analysis, fusion) and hands all outputs to the
new decision engine for confidence, mode, reasoning, and explanation.

Usage (from num_pipeline/scripts/):
    python run_diagnostic.py "brake pedal feels spongy"
    python run_diagnostic.py "engine overheating"
    python run_diagnostic.py --no-sensor "check engine light"
    python run_diagnostic.py --use-llm -v "check engine light"
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Ensure num_pipeline/scripts/ is in path ──
SCRIPTS_DIR = Path(__file__).resolve().parent
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── Change to num_pipeline dir for sensor_analysis relative paths ──
os.chdir(NUM_PIPELINE_DIR)

# ── Existing pipeline modules (unchanged) ──
from pipeline.query_preprocessor import preprocess_query
from pipeline.hybrid_retrieval import hybrid_retrieve
from pipeline.fault_mapper import map_faults
from pipeline.evidence_fusion import fuse_evidence
from sensor_validation.sensor_analysis import analyze_fault_candidates
import pandas as pd
from pipeline.llm_provider import get_llm_provider

# ── New decision engine ──
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from num_pipeline.scripts.decision_engine.engine import (
    run_diagnostic_engine,
    DiagnosticReport,
)


def run_diagnostic(
    symptoms_text: str = None,
    current_sample=None,
    use_llm: bool = True,
    speed: int = 1000,
    verbose: bool = False,
) -> DiagnosticReport:
    """Run the full diagnostic pipeline with the new decision engine.

    Parameters
    ----------
    symptoms_text : str
        Free-text symptom description.
    current_sample : dict, "simulated", or None
        Sensor data for validation.  When a dict (parsed ECU CSV row),
        it is used directly.  When the string ``"simulated"``, a
        representative row is loaded from the FAULT CSV matching the
        top mapped navic_fault.  When None, sensor analysis is skipped.
    use_llm : bool
        Attempt to use LLM for explanations (default True).
        When True, the LLM provider is probed; if Ollama is
        unavailable the engine falls back to deterministic
        templates automatically.
    speed : int
        Engine speed for sensor profile and CSV selection (1000, 1200,
        1400, or 1600).
    verbose : bool
        Print detailed output.

    Returns
    -------
    DiagnosticReport
        Single output object consumed by the UI.
    """
    if not symptoms_text:
        raise ValueError("Provide symptoms_text")

    # ── Step 1: Query Preprocessing ───────────────────────────────
    if verbose:
        print("=" * 60)
        print("STEP 1: Query Preprocessing")
        print("=" * 60)

    preprocessed = preprocess_query(symptoms_text)

    if verbose:
        print(f"  Processed: {preprocessed['processed']}")
        print(f"  Entities: {[e['label'] for e in preprocessed['entities']]}")

    # ── Step 2: Hybrid Retrieval ──────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Hybrid Retrieval")
        print("=" * 60)

    retrieval_result = hybrid_retrieve(symptoms_text, top_k=10)

    if verbose:
        n = len(retrieval_result["candidates"])
        print(f"  Found {n} candidates")

    # ── Step 3: Fault Mapping ─────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Fault Mapping")
        print("=" * 60)

    mapped_faults = map_faults(retrieval_result)

    if verbose:
        print(f"  Mapped {len(mapped_faults)} faults")

    # ── Step 4: Sensor Analysis (optional) ────────────────────────
    sensor_results = {}
    sensor_sample = None
    if current_sample is not None:
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 4: Sensor Analysis")
            print("=" * 60)

        try:
            sensor_sample = current_sample

            # "simulated" sentinel → load a representative row from
            # the FAULT CSV matching the top mapped navic_fault
            if sensor_sample == "simulated":
                if mapped_faults:
                    navic_fault = mapped_faults[0].get("navic_fault", "")
                    csv_path = (
                        NUM_PIPELINE_DIR / "data" / "processed"
                        / f"INCA_SPEED_{speed}_{navic_fault}.csv"
                    )
                    if verbose:
                        print(f"  Simulated sample from: {csv_path}")
                    df = pd.read_csv(csv_path)
                    sensor_sample = df.sample(1).iloc[0].to_dict()
                else:
                    if verbose:
                        print("  No mapped faults — skipping sensor analysis")
                    sensor_sample = None

            if sensor_sample is not None:
                sensor_results = analyze_fault_candidates(
                    mapped_faults=mapped_faults,
                    speed=speed,
                    current_sample=sensor_sample,
                )
                if verbose:
                    for fault, result in sensor_results.items():
                        print(f"  {fault}: confidence={result['sensor_confidence']:.2f}")
        except Exception as e:
            if verbose:
                print(f"  Sensor analysis failed: {e}")
            sensor_results = {}

    # ── Sensor debug metadata ─────────────────────────────────────
    sensor_debug = {
        "ran": bool(sensor_results),
        "input_type": (
            "simulated" if current_sample == "simulated"
            else "uploaded" if isinstance(current_sample, dict)
            else "none"
        ),
        "speed": speed,
        "navic_fault": mapped_faults[0].get("navic_fault", "") if mapped_faults else "",
        "csv_path": "",
        "sample_columns": [],
    }
    if current_sample == "simulated" and mapped_faults:
        nf = mapped_faults[0].get("navic_fault", "")
        sensor_debug["csv_path"] = f"INCA_SPEED_{speed}_{nf}.csv"
    if isinstance(sensor_sample, dict):
        sensor_debug["sample_columns"] = sorted(sensor_sample.keys())[:8]
    elif isinstance(current_sample, dict):
        sensor_debug["sample_columns"] = sorted(current_sample.keys())[:8]

    # ── Step 5: Evidence Fusion ───────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: Evidence Fusion")
        print("=" * 60)

    fused_result = fuse_evidence(retrieval_result, mapped_faults, sensor_results)

    if verbose:
        print(f"  Fused {len(fused_result['fused_candidates'])} candidates")

    # ── Step 6: Decision Engine ───────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 6: Decision Engine")
        print("=" * 60)

    llm_provider = None
    if use_llm:
        try:
            candidate = get_llm_provider()
            if candidate.is_available():
                llm_provider = candidate
                if verbose:
                    print("  LLM: available")
            elif verbose:
                print("  LLM: not available (falling back to templates)")
        except Exception:
            if verbose:
                print("  LLM: probe failed (falling back to templates)")

    report = run_diagnostic_engine(
        preprocessed=preprocessed,
        retrieval_result=retrieval_result,
        mapped_faults=mapped_faults,
        sensor_results=sensor_results,
        fused_result=fused_result,
        llm_provider=llm_provider,
    )
    report.sensor_debug = sensor_debug

    if verbose:
        print(f"  Mode: {report.mode}")
        print(f"  Confidence: {report.confidence:.3f}")
        print(f"  Top candidate: {report.top_candidate.get('label', 'None')}")
        print(f"  Display candidates: {len(report.display_candidates)}")
        print(f"  Summary: {report.summary}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle fault diagnosis (new decision engine)"
    )
    parser.add_argument("symptoms", nargs="?", help="Symptom description")
    parser.add_argument("--no-sensor", action="store_true",
                        help="Skip sensor analysis (no simulated or uploaded data)")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--speed", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON output")

    args = parser.parse_args()

    if not args.symptoms:
        parser.print_help()
        return

    sensor_arg = None if args.no_sensor else "simulated"

    report = run_diagnostic(
        symptoms_text=args.symptoms,
        current_sample=sensor_arg,
        use_llm=args.use_llm,
        speed=args.speed,
        verbose=args.verbose,
    )

    if args.json:
        output = {
            "mode": report.mode,
            "confidence": report.confidence,
            "confidence_components": report.confidence_components,
            "top_candidate": {
                "label": report.top_candidate.get("label", ""),
                "confidence": report.top_candidate.get("confidence", 0.0),
                "navic_fault": report.top_candidate.get("navic_fault", ""),
                "mapping_type": report.top_candidate.get("mapping_type", ""),
                "sensor_status": report.top_candidate.get("sensor_status", ""),
            },
            "display_candidates": [
                {
                    "label": c.get("label", ""),
                    "confidence": c.get("confidence", 0.0),
                    "navic_fault": c.get("navic_fault", ""),
                    "mapping_type": c.get("mapping_type", ""),
                    "sensor_status": c.get("sensor_status", ""),
                }
                for c in report.display_candidates
            ],
            "reasoning_chain": report.reasoning_chain,
            "sensor_evidence": report.sensor_evidence,
            "summary": report.summary,
            "original_symptoms": report.original_symptoms,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Mode: {report.mode}")
        print(f"Confidence: {report.confidence:.0%}")
        print(f"{'='*60}")
        print(report.summary)
        print()
        print(report.explanation)


if __name__ == "__main__":
    main()
