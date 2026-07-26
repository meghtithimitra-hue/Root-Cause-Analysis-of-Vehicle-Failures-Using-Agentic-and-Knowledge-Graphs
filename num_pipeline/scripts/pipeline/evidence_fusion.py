from typing import Dict
import copy

# ==========================================================
# Evidence Weights
# ==========================================================

KG_WEIGHT = 0.45

SENSOR_WEIGHT = 0.35

CRITICAL_WEIGHT = 0.15

WARNING_WEIGHT = 0.05


# ==========================================================
# Normalize score
# ==========================================================

def normalize(score, minimum=0.0, maximum=2.0):

    score = max(minimum, min(score, maximum))

    return (score - minimum) / (maximum - minimum)

def compute_sensor_bonus(sensor_result):

    critical = len(sensor_result.get("critical", []))

    warning = len(sensor_result.get("warning", []))

    normal = len(sensor_result.get("normal", []))

    total = max(
        critical + warning + normal,
        1
    )

    critical_score = critical / total

    warning_score = warning / total

    return critical_score, warning_score


# ==========================================================
# Fuse one candidate
# ==========================================================

def fuse_candidate(candidate, sensor_result):

    kg_score = normalize(candidate["score"])

    sensor_score = sensor_result.get(
        "sensor_confidence",
        0
    )

    critical_score, warning_score = compute_sensor_bonus(
        sensor_result
    )

    mapping_score = candidate.get(

        "mapping_confidence",

        1.0

    )

    final_score = (

        0.45 * kg_score +

        0.20 * mapping_score +

        0.35 * sensor_score

    )

    candidate = copy.deepcopy(candidate)

    candidate["kg_score"] = round(kg_score, 3)

    candidate["sensor_score"] = round(sensor_score, 3)

    candidate["critical_score"] = round(
        critical_score,
        3
    )

    candidate["warning_score"] = round(
        warning_score,
        3
    )

    candidate["final_score"] = round(
        final_score,
        3
    )

    candidate["critical_sensors"] = sensor_result.get(
        "critical",
        []
    )

    candidate["warning_sensors"] = sensor_result.get(
        "warning",
        []
    )

    candidate["normal_sensors"] = sensor_result.get(
        "normal",
        []
    )
    print("\nFUSION")

    print(candidate["label"])

    print("KG:", kg_score)

    print("Sensor:", sensor_score)

    print("Mapping:", mapping_score)

    print("Final:", final_score)
    return candidate

# ==========================================================
# Fuse all candidates
# ==========================================================

def fuse_evidence(

        retrieval_result,

        mapped_faults,

        sensor_results

):

    fused = []

    # --------------------------------------------------
    # KG label -> list of mapped Navic faults
    # --------------------------------------------------

    label_to_faults = {}

    for m in mapped_faults:

        label = m["label"]

        if label not in label_to_faults:

            label_to_faults[label] = []

        label_to_faults[label].append(

            m["navic_fault"]

        )
    print("\n========== LABEL TO FAULTS ==========")

    for label, faults in label_to_faults.items():

        print(label)

        print(faults)

    # --------------------------------------------------
    # Fuse every KG candidate
    # --------------------------------------------------

    for candidate in retrieval_result["candidates"]:

        label = candidate["label"]

        mapped = label_to_faults.get(

            label,

            []

        )
        print("\nCandidate:")

        print(label)

        print("Mapped faults:")

        print(mapped)

        best_sensor = {

            "sensor_confidence":0,

            "critical":[],

            "warning":[],

            "normal":[]

        }

        best_score = 0

        # ----------------------------------------------
        # Compare all mapped Navic faults
        # ----------------------------------------------

        for fault in mapped:

            result = sensor_results.get(fault)

            if result is None:

                continue

            score = result.get(

                "sensor_confidence",

                0

            )

            if score > best_score:

                best_score = score

                best_sensor = result
            print(
                f"\n{label}"
            )

            print(
                "Mapped faults:",
                mapped
            )

            print(
                "Best sensor score:",
                best_score
            )
            print(

            fault,

            sensor_results[fault]["sensor_confidence"]

            )
        fused.append(

            fuse_candidate(

                candidate,

                best_sensor

            )

        )

    fused.sort(

        key=lambda x:x["final_score"],

        reverse=True

    )

    return {

        "query":retrieval_result["query"],

        "fused_candidates":fused

    }

# ==========================================================
# Pretty print
# ==========================================================

def print_results(result):

    print()

    print("=" * 110)

    print("FINAL DIAGNOSIS RANKING")

    print("=" * 110)

    print(

        f"{'Rank':<5}"

        f"{'Final':<8}"

        f"{'KG':<8}"

        f"{'Sensor':<10}"

        f"{'Critical':<10}"

        f"{'Warning':<10}"

        f"{'Fault'}"

    )

    print("-" * 110)

    for i, c in enumerate(result["fused_candidates"], 1):

        print(

            f"{i:<5}"

            f"{c['final_score']:<8.3f}"

            f"{c['kg_score']:<8.3f}"

            f"{c['sensor_score']:<10.3f}"

            f"{c['critical_score']:<10.3f}"

            f"{c['warning_score']:<10.3f}"

            f"{c['label']}"

        )