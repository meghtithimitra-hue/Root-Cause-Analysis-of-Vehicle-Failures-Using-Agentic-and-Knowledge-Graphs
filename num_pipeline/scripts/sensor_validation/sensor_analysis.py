from pathlib import Path
import json
import numpy as np

# ==========================================================
# Configuration
# ==========================================================

PROFILE_DIR = Path("outputs/profiles")
MAPPING_FILE = Path("data/processed/sensor_mapping.json")


# ==========================================================
# Load Sensor Mapping
# ==========================================================

with open(MAPPING_FILE, "r") as f:
    SENSOR_MAPPING = json.load(f)


# ==========================================================
# Load Profile
# ==========================================================

def load_profile(speed, condition):
    """
    Load profile JSON.

    Example:
        speed = 1000
        condition = "NOMINAL"

    Loads:
        outputs/profiles/INCA_SPEED_1000_NOMINAL.json
    """

    profile_file = (
        PROFILE_DIR /
        f"INCA_SPEED_{speed}_{condition}.json"
    )

    if not profile_file.exists():

        print(f"Profile not found: {profile_file}")

        return None

    with open(profile_file, "r") as f:

        profile = json.load(f)

    return profile


# ==========================================================
# Load Sensors for a Fault
# ==========================================================

def get_fault_sensors(fault, speed):
    """
    Reads sensor_mapping.json and returns
    only the sensor names.

    Returns:

    [
        "amp_mes",
        "prs_cmpr_up",
        ...
    ]
    """

    if fault not in SENSOR_MAPPING:

        return []

    if str(speed) not in SENSOR_MAPPING[fault]:

        return []

    sensor_block = SENSOR_MAPPING[fault][str(speed)]

    sensors = [

        sensor["sensor"]

        for sensor in sensor_block["sensors"]

    ]

    return sensors


# ==========================================================
# Z-score
# ==========================================================

def compute_z_score(value, mean, std):

    if std == 0:

        return 0.0

    return (value - mean) / std


# ==========================================================
# Percentage Change
# ==========================================================

def percent_change(current, nominal):

    if nominal == 0:

        return 0.0

    return ((current - nominal) / nominal) * 100


# ==========================================================
# Sensor Status
# ==========================================================

# ==========================================================
# Sensor Status Classification
# ==========================================================

def classify_sensor(
    z_score,
    percent_change,
    sensor_confidence
):
    """
    Classify sensor health using:
        1. Statistical deviation (z-score)
        2. Engineering deviation (% change)
        3. Similarity to fault profile

    Returns:
        NORMAL
        WARNING
        CRITICAL
    """

    abs_z = abs(z_score)
    abs_pct = abs(percent_change)

    # Strong evidence of fault
    if (
        sensor_confidence >= 0.90
        and abs_z >= 5
    ):
        return "CRITICAL"

    # Moderate evidence
    elif (
        sensor_confidence >= 0.70
        and abs_z >= 3
    ):
        return "WARNING"

    # Large engineering deviation
    elif abs_pct >= 10:
        return "WARNING"

    else:
        return "NORMAL"

# ==========================================================
# Sensor Confidence
# ==========================================================

def compute_sensor_similarity(
        value,
        nominal_mean,
        fault_mean
):
    """
    Computes similarity between the current sensor value
    and the fault profile.

    Uses exponential decay so confidence decreases
    smoothly instead of abruptly becoming zero.
    """

    distance = abs(value - fault_mean)

    scale = max(
        abs(fault_mean),
        abs(nominal_mean),
        1e-6
    )

    similarity = np.exp(
        -distance / (0.05 * scale)
    )

    return round(float(similarity), 3)
# ==========================================================
# Analyse One Fault
# ==========================================================

def analyze_signals(
        fault,
        speed,
        current_sample
):
    """
    Analyse one candidate NavicEngine fault.

    Parameters
    ----------
    fault : str
        Example:
            FAULT_INJ_DUR

    speed : int

    current_sample : dict
        Example:

        {
            "amp_mes":18.7,
            "prs_cmpr_up":220,
            ...
        }

    Returns
    -------
    Dictionary containing sensor evidence.
    """

    # ------------------------------------------------------
    # Load Profiles
    # ------------------------------------------------------

    nominal_profile = load_profile(
        speed,
        "NOMINAL"
    )

    if nominal_profile is None:

        return None

    fault_profile = load_profile(
        speed,
        fault
    )

    if fault_profile is None:

        return None

    # ------------------------------------------------------
    # Load mapped sensors
    # ------------------------------------------------------

    sensors = get_fault_sensors(
        fault,
        speed
    )

    signal_results = []

    confidence_scores = []

    importance_weights = []

    # ------------------------------------------------------
    # Analyse every selected sensor
    # ------------------------------------------------------

    for sensor in sensors:

        if sensor not in current_sample:
            continue

        if sensor not in nominal_profile:
            continue

        if sensor not in fault_profile:
            continue

        value = current_sample[sensor]

        nominal_mean = nominal_profile[sensor]["mean"]
        nominal_std = nominal_profile[sensor]["std"]

        fault_mean = fault_profile[sensor]["mean"]

        # -------------------------------
        # Statistics
        # -------------------------------

        z = compute_z_score(
            value,
            nominal_mean,
            nominal_std
        )

        pct = percent_change(
            value,
            nominal_mean
        )

        similarity = compute_sensor_similarity(
            value,
            nominal_mean,
            fault_mean
        )

        status = classify_sensor(
            z_score=z,
            percent_change=pct,
            sensor_confidence=similarity
        )

        # Read importance score from sensor_mapping.json
        importance = 1.0

        sensor_info = SENSOR_MAPPING[fault][str(speed)]["sensors"]

        for item in sensor_info:

            if item["sensor"] == sensor:

                importance = item.get(
                    "importance_score",
                    1.0
                )

                break

        confidence_scores.append(similarity)
        importance_weights.append(importance)
        signal_results.append({

            "sensor": sensor,

            "current_value": round(value, 4),

            "nominal_mean": round(
                nominal_mean,
                4
            ),

            "fault_mean": round(
                fault_mean,
                4
            ),

            "z_score": round(z, 3),

            "percent_change": round(
                pct,
                2
            ),

            "status": status,

            "sensor_confidence": round(
                similarity,
                3
            )

        })

    # ------------------------------------------------------
    # Overall Sensor Confidence
    # ------------------------------------------------------

    if confidence_scores:

        final_confidence = float(

            np.average(

                confidence_scores,

                weights=importance_weights

            )

        )

    else:

        final_confidence = 0.0

    # ------------------------------------------------------
    # Categorize Sensors
    # ------------------------------------------------------

    critical = [

        s["sensor"]

        for s in signal_results

        if s["status"] == "CRITICAL"

    ]

    warning = [

        s["sensor"]

        for s in signal_results

        if s["status"] == "WARNING"

    ]

    normal = [

        s["sensor"]

        for s in signal_results

        if s["status"] == "NORMAL"

    ]

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return {

        "fault": fault,

        "speed": speed,

        "num_sensors_checked": len(
            signal_results
        ),

        "critical": critical,

        "warning": warning,

        "normal": normal,

        "sensor_confidence": round(
            final_confidence,
            3
        ),

        "sensor_results": signal_results

    }
# ==========================================================
# Analyse Multiple Candidate Faults
# ==========================================================

def analyze_fault_candidates(
        mapped_faults,
        speed,
        current_sample
):
    """
    Analyse all candidate faults returned by fault_mapper.py.

    Parameters
    ----------
    mapped_faults : list

    Example

    [
        {
            "kg_category":"Engine System",
            "navic_fault":"FAULT_INJ_DUR",
            "kg_score":1.82
        },

        {
            "kg_category":"Engine System",
            "navic_fault":"FAULT_INJ_PRS",
            "kg_score":1.74
        }
    ]

    speed : int

    current_sample : dict

    Returns
    -------

    Dictionary indexed by NavicEngine fault.

    {
        "FAULT_INJ_DUR": {...},

        "FAULT_INJ_PRS": {...}
    }
    """

    results = {}

    for candidate in mapped_faults:

        fault = candidate["navic_fault"]

        result = analyze_signals(

            fault=fault,

            speed=speed,

            current_sample=current_sample

        )

        if result is None:
            continue

        # Preserve KG metadata for evidence fusion

        result["kg_category"] = candidate.get(
            "kg_category",
            "Unknown"
        )

        result["kg_score"] = candidate.get(
            "kg_score",
            0.0
        )

        result["kg_label"] = candidate.get(
            "label",
            ""
        )

        result["node_type"] = candidate.get(
            "node_type",
            ""
        )

        result["source"] = candidate.get(
            "source",
            ""
        )

        results[fault] = result

    return results


# ==========================================================
# Pretty Print Results
# ==========================================================

def print_results(results):

    print("\n")

    print("=" * 80)

    print("NAVIC SENSOR ANALYSIS")

    print("=" * 80)

    for fault, result in results.items():

        print(f"\nFault : {fault}")

        print(f"KG Category : {result['kg_category']}")

        print(f"KG Score : {result['kg_score']:.3f}")

        print(
            f"Sensor Confidence : "
            f"{result['sensor_confidence']:.3f}"
        )

        print(
            f"Sensors Checked : "
            f"{result['num_sensors_checked']}"
        )

        print(
            f"Critical : "
            f"{', '.join(result['critical']) if result['critical'] else 'None'}"
        )

        print(
            f"Warning : "
            f"{', '.join(result['warning']) if result['warning'] else 'None'}"
        )

        print("-" * 80)


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(
    "data/processed/INCA_SPEED_1000_FAULT_INJ_DUR.csv"
     )
    mapped_faults = [

        {
            "kg_category": "Engine System",

            "navic_fault": "FAULT_INJ_DUR",

            "kg_score": 1.82,

            "label": "Poor Engine Performance",

            "node_type": "Symptom",

            "source": "Hybrid Retrieval"
        },

        {
            "kg_category": "Engine System",

            "navic_fault": "FAULT_INJ_PRS",

            "kg_score": 1.74,

            "label": "High Fuel Consumption",

            "node_type": "Symptom",

            "source": "Hybrid Retrieval"
        }

    ]

    sample = df.sample(1).iloc[0].to_dict()

    results = analyze_fault_candidates(

        mapped_faults=mapped_faults,

        speed=1000,

        current_sample=sample

    )

    print_results(results)

    print("\n")

    print(json.dumps(results, indent=4))