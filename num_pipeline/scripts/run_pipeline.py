"""
run_pipeline.py

Complete Vehicle Fault Diagnosis Pipeline

Pipeline:

User Query
    ↓
Query Preprocessor
    ↓
Hybrid Retrieval
    ↓
Sensor Validation
    ↓
Evidence Fusion
    ↓
Final Diagnosis
"""

from pipeline.query_preprocessor import preprocess_query
from pipeline.hybrid_retrieval import hybrid_retrieve
from pipeline.evidence_fusion import fuse_evidence
from pipeline.fault_mapper import map_faults
from sensor_validation.sensor_analysis import analyze_fault_candidates
from sensor_validation.current_sample import load_current_sensor_sample
import pandas as pd
# ==========================================================
# Run complete pipeline
# ==========================================================

def run_pipeline(query, speed=1000):

    print("\n" + "=" * 90)
    print("VEHICLE FAULT DIAGNOSIS PIPELINE")
    print("=" * 90)

    # ------------------------------------------------------
    # Step 1 : Query preprocessing
    # ------------------------------------------------------

    print("\n[1] Query Preprocessing...")

    preprocessed = preprocess_query(query)


    processed_query = preprocessed["processed"]
    retrieval_hints = preprocessed["retrieval_hints"]

    expected_sensors = preprocessed["expected_sensors"]

    error_codes = preprocessed["error_codes"]

    print("Processed Query :", processed_query)
    print("Communities     :", retrieval_hints["communities"])
    print("Categories      :", retrieval_hints["categories"])
    print("ExpectedSensors :", expected_sensors)
    print("Error Codes     :", error_codes)

    # ------------------------------------------------------
    # Step 2 : Hybrid Retrieval
    # ------------------------------------------------------

    print("\n[2] Hybrid Retrieval...")

    retrieval_result = hybrid_retrieve(
        query,
        top_k=5
    )

    print("\nRetrieved Fault Candidates\n")

    for i, c in enumerate(
            retrieval_result["candidates"],
            1):

        print(
            f"{i}. {c['label']} "
            f"({c['score']:.3f})"
        )

    # ------------------------------------------------------
    # Step 3 : Fault Mapping
    # ------------------------------------------------------

    print("\n[3] Fault Mapping...")

    mapped_faults = map_faults(retrieval_result)
    print("\n========== MAPPED FAULTS ==========")

    for m in mapped_faults:

        print(
            m["label"],
            "->",
            m["navic_fault"]
        )

    for m in mapped_faults:
        print(
            f"{m['navic_fault']}  ({m['kg_category']})"
        )


    # ------------------------------------------------------
    # Step 4 : Sensor Validation
    # ------------------------------------------------------

    print("\n[4] Sensor Validation...")

    # Load one processed NavicEngine sample
    df = pd.read_csv(
        "data/processed/INCA_SPEED_1000_FAULT_INJ_DUR.csv"
    )

    # Temporary test sample until live sensor data is available
    # ------------------------------------------------------
        # Current Sensor Readings
        # ------------------------------------------------------

        # TODO:
        # Replace this with live ECU data or
        # a simulator in the future.

    current_sample = load_current_sensor_sample(speed)
            
    sensor_results = analyze_fault_candidates(

        mapped_faults=mapped_faults,

        speed=speed,

        current_sample=current_sample

    )
    print("\n========== SENSOR RESULTS ==========")

    for fault, result in sensor_results.items():

        print(f"\nFault: {fault}")

        print("Sensor confidence:",
            result["sensor_confidence"])

        print("Critical:",
            len(result["critical"]))

        print("Warning:",
            len(result["warning"]))

        print("Normal:",
            len(result["normal"]))
    # ------------------------------------------------------
    # Step 4 : Evidence Fusion
    # ------------------------------------------------------

    print("\n[4] Evidence Fusion...")

    fused_result = fuse_evidence(

        retrieval_result,

        mapped_faults,

        sensor_results

    )

    # ------------------------------------------------------
    # Step 5 : Final Ranking
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL DIAGNOSIS")
    print("=" * 90)

    for i, candidate in enumerate(

            fused_result["fused_candidates"],

            1):

        print(f"\nRank {i}")

        print("-" * 60)

        print("Fault :", candidate["label"])

        print("Final Score :", candidate["final_score"])

        print("KG Score :", candidate["kg_score"])

        print("Sensor Score :", candidate["sensor_score"])

        print("Source :", candidate["source"])

        print("Critical Sensors :",
              candidate["critical_sensors"])

        print("Warning Sensors :",
              candidate["warning_sensors"])

        print("Normal Sensors :",
              candidate["normal_sensors"])

    return fused_result


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    queries = [

        "poor engine performance",

        "high fuel consumption",

        "fuel injector problem",

        "injector malfunction",

        "engine misfire",

        "low engine power",

        "engine hesitation",

        "engine consumes too much fuel"

    ]

    for q in queries:

        run_pipeline(

            query=q,

            speed=1000

        )