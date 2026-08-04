"""
run_pipeline.py — End-to-end diagnostic pipeline.

Integrates all modules into a single diagnostic flow:
1. Query Preprocessing
2. Hybrid Retrieval
3. Fault Mapping
4. Sensor Analysis (optional)
5. Evidence Fusion
6. Reasoning Engine
7. LLM Explanation

Usage (from num_pipeline/scripts/):
    python run_pipeline.py "brake pedal feels spongy"
    python run_pipeline.py --skip-sensor "engine overheating"
    python run_pipeline.py --use-llm -v "check engine light"
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Ensure num_pipeline/scripts/ is in path (for existing imports) ──
SCRIPTS_DIR = Path(__file__).resolve().parent
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── Change to num_pipeline dir for sensor_analysis relative paths ──
os.chdir(NUM_PIPELINE_DIR)

from pipeline.query_preprocessor import preprocess_query
from pipeline.hybrid_retrieval import hybrid_retrieve
from pipeline.fault_mapper import map_faults
from pipeline.evidence_fusion import fuse_evidence
from sensor_validation.sensor_analysis import analyze_fault_candidates
from sensor_validation.current_sample import load_current_sensor_sample

# ── New modules (from project root) ────────────────────────────────
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from num_pipeline.scripts.pipeline.reasoning_engine import reason, format_reasoning_chain
from num_pipeline.scripts.pipeline.explanation_generator import (
    generate_explanation, generate_brief_summary
)
from num_pipeline.scripts.pipeline.llm_provider import get_llm_provider


def run_diagnostic(
    symptoms_text: str = None,
    skip_sensor: bool = False,
    use_llm: bool = False,
    speed: int = 1000,
    verbose: bool = False
) -> dict:
    """
    Run the full diagnostic pipeline.

    Args:
        symptoms_text: Free-text symptom description
        skip_sensor: Skip sensor analysis
        use_llm: Use LLM for explanation (if available)
        speed: Vehicle speed for sensor profiles
        verbose: Print detailed output

    Returns:
        {
            "decision": DiagnosticDecision,
            "evidence": dict,
            "explanation": str,
            "summary": str,
            "fused_candidates": list
        }
    """

    if not symptoms_text:
        raise ValueError("Provide symptoms_text")

    # ── Step 1: Query Preprocessing ───────────────────────────────

    if verbose:
        print("=" * 60)
        print("STEP 1: Query Preprocessing")
        print("=" * 60)

    preprocessed = preprocess_query(symptoms_text)
    processed_query = preprocessed["processed"]
    retrieval_hints = preprocessed["retrieval_hints"]
    expected_sensors = preprocessed["expected_sensors"]

    if verbose:
        print(f"  Processed: {processed_query}")
        print(f"  Communities: {retrieval_hints['communities']}")
        print(f"  Categories: {retrieval_hints['categories']}")
        print(f"  Expected sensors: {expected_sensors}")

    # ── Step 2: Hybrid Retrieval ──────────────────────────────────

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Hybrid Retrieval")
        print("=" * 60)

    retrieval_result = hybrid_retrieve(symptoms_text, top_k=10)

    if verbose:
        print(f"  Found {len(retrieval_result['candidates'])} candidates")
        stats = retrieval_result["retrieval_stats"]
        print(f"  Vector: {stats['vector_candidates']}, "
              f"Graph: {stats['graph_candidates']}, "
              f"Community: {stats['community_candidates']}")

    # ── Step 3: Fault Mapping ─────────────────────────────────────

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Fault Mapping")
        print("=" * 60)

    mapped_faults = map_faults(retrieval_result)

    if verbose:
        print(f"  Mapped {len(mapped_faults)} faults")
        for m in mapped_faults[:3]:
            print(f"  {m['label']} -> {m['navic_fault']}")

    # ── Step 4: Sensor Analysis (optional) ────────────────────────

    sensor_results = {}
    if not skip_sensor:
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 4: Sensor Analysis")
            print("=" * 60)

        try:
            current_sample = load_current_sensor_sample(speed)
            sensor_results = analyze_fault_candidates(
                mapped_faults=mapped_faults,
                speed=speed,
                current_sample=current_sample
            )
            if verbose:
                for fault, result in sensor_results.items():
                    print(f"  {fault}: confidence={result['sensor_confidence']:.2f}, "
                          f"critical={len(result['critical'])}, "
                          f"warning={len(result['warning'])}")
        except Exception as e:
            if verbose:
                print(f"  Sensor analysis failed: {e}")
            sensor_results = {}

    # ── Step 5: Evidence Fusion ───────────────────────────────────

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: Evidence Fusion")
        print("=" * 60)

    fused_result = fuse_evidence(retrieval_result, mapped_faults, sensor_results)

    if verbose:
        print(f"  Fused {len(fused_result['fused_candidates'])} candidates")
        for c in fused_result["fused_candidates"][:3]:
            print(f"  {c['label']}: final={c['final_score']:.3f}, "
                  f"kg={c['kg_score']:.3f}, sensor={c['sensor_score']:.3f}")

    # ── Step 6: Reasoning Engine ──────────────────────────────────

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 6: Reasoning Engine")
        print("=" * 60)

    # Convert fused_result to format expected by reasoning engine
    evidence = _prepare_evidence_for_reasoning(
        fused_result, sensor_results, preprocessed
    )
    decision = reason(evidence)

    if verbose:
        print(f"  Mode: {decision.mode}")
        print(f"  Reason: {decision.mode_reason}")
        print(f"  Predicted: {decision.predicted_system} > {decision.predicted_subsystem}")
        print(f"  Evidence quality: {decision.evidence_quality}")
        print("\n  Reasoning Chain:")
        print(format_reasoning_chain(decision.reasoning_chain))

    # ── Step 7: LLM Explanation ───────────────────────────────────

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 7: Explanation Generation")
        print("=" * 60)

    llm_provider = None
    if use_llm:
        llm_provider = get_llm_provider()

    explanation = generate_explanation(decision, evidence, llm_provider)
    summary = generate_brief_summary(decision, evidence)

    if verbose:
        print(f"  Explanation length: {len(explanation)} chars")
        print(f"  Summary: {summary}")

    return {
        "decision": decision,
        "evidence": evidence,
        "explanation": explanation,
        "summary": summary,
        "fused_candidates": fused_result["fused_candidates"]
    }


def _prepare_evidence_for_reasoning(fused_result, sensor_results, preprocessed):
    """
    Convert fused_result to the format expected by reasoning engine.

    The reasoning engine expects:
    {
        "candidate_faults": [...],
        "sensor_evidence": {...},
        "kg_evidence": [...],
        "sensor_status": str,
        "kg_status": str,
        "final_score": float,
        "matched_symptoms": [...],
        "missing_symptoms": [...]
    }
    """
    candidates = fused_result.get("fused_candidates", [])

    # Convert to format expected by reasoning engine
    candidate_faults = []
    for c in candidates:
        # Extract system/subsystem from category/subcategory or label
        label = c.get("label", "Unknown")
        category = c.get("category", c.get("kg_category", ""))
        subcategory = c.get("subcategory", "")

        # Determine system and subsystem
        system = category if category else "Unknown"
        subsystem = subcategory if subcategory else label

        candidate_faults.append({
            "system": system,
            "subsystem": subsystem,
            "label": label,
            "final_score": c.get("final_score", 0.0),
            "kg_score": c.get("kg_score", 0.0),
            "sensor_score": c.get("sensor_score", 0.0),
            "matched_symptoms": [label],  # Use label as proxy for matched symptom
            "sensor_status": _get_sensor_status(c, sensor_results)
        })

    # Determine sensor status
    sensor_status = "NOT AVAILABLE"
    if sensor_results:
        # Check if any sensor data exists
        has_sensors = any(
            result.get("sensor_confidence", 0) > 0
            for result in sensor_results.values()
        )
        if has_sensors:
            # Check if sensors confirm or contradict
            confirmed = any(
                result.get("sensor_confidence", 0) > 0.5
                for result in sensor_results.values()
            )
            sensor_status = "CONFIRMS" if confirmed else "NOT AVAILABLE"

    # Determine KG status
    kg_status = "NOT AVAILABLE"
    if candidates:
        top_score = candidates[0].get("final_score", 0.0)
        if top_score >= 0.7:
            kg_status = "CONFIRMS"
        elif top_score < 0.3:
            kg_status = "CONTRADICTS"

    # Get symptoms from preprocessed query
    matched_symptoms = preprocessed.get("expanded_queries", [])
    if not matched_symptoms:
        matched_symptoms = [fused_result.get("query", "")]

    return {
        "candidate_faults": candidate_faults,
        "sensor_evidence": sensor_results if sensor_results else None,
        "kg_evidence": candidate_faults,
        "sensor_status": sensor_status,
        "kg_status": kg_status,
        "final_score": candidates[0].get("final_score", 0.0) if candidates else 0.0,
        "matched_symptoms": matched_symptoms,
        "missing_symptoms": []
    }


def _get_sensor_status(candidate, sensor_results):
    """Get sensor status for a specific candidate."""
    navic_fault = candidate.get("navic_fault", "")
    if navic_fault in sensor_results:
        result = sensor_results[navic_fault]
        confidence = result.get("sensor_confidence", 0)
        if confidence > 0.5:
            return "CONFIRMED"
        elif confidence > 0:
            return "NOT AVAILABLE"
    return "NOT AVAILABLE"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vehicle Fault Diagnosis Pipeline"
    )
    parser.add_argument(
        "symptoms",
        nargs="?",
        help="Symptom description (free text)"
    )
    parser.add_argument(
        "--skip-sensor",
        action="store_true",
        help="Skip sensor analysis"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for explanation (if available)"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1000,
        help="Vehicle speed for sensor profiles (default: 1000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed pipeline output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    if not args.symptoms:
        parser.print_help()
        sys.exit(1)

    result = run_diagnostic(
        symptoms_text=args.symptoms,
        skip_sensor=args.skip_sensor,
        use_llm=args.use_llm,
        speed=args.speed,
        verbose=args.verbose
    )

    if args.json:
        output = {
            "mode": result["decision"].mode,
            "mode_reason": result["decision"].mode_reason,
            "predicted_system": result["decision"].predicted_system,
            "predicted_subsystem": result["decision"].predicted_subsystem,
            "evidence_quality": result["decision"].evidence_quality,
            "sensor_status": result["evidence"]["sensor_status"],
            "kg_status": result["evidence"]["kg_status"],
            "matched_symptoms": result["evidence"]["matched_symptoms"],
            "missing_symptoms": result["evidence"]["missing_symptoms"],
            "candidate_count": len(result["evidence"]["candidate_faults"]),
            "reasoning_chain": result["decision"].reasoning_chain,
            "confirmed_faults": result["decision"].confirmed_faults,
            "contradicted_faults": result["decision"].contradicted_faults,
            "remaining_faults": result["decision"].remaining_faults,
            "explanation": result["explanation"],
            "summary": result["summary"]
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + result["explanation"])
        print("\n" + "=" * 60)
        print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
