from pathlib import Path
import pandas as pd
import json
import re

# ==========================================================
# Configuration
# ==========================================================

COMPARISON_DIR = Path("outputs/comparisons")
OUTPUT_FILE = Path("data/processed/sensor_mapping.json")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TOP_K = 10
PVALUE_THRESHOLD = 0.05
MIN_EFFECT = 0.8

# Ignore non-diagnostic signals
IGNORE = {
    "time",
    "timestamp",
    "index"
}


# ==========================================================
# Parse filename
# Example:
# 1000_FAULT_INJ_DUR_vs_NOMINAL.csv
# ==========================================================

def parse_filename(filename):

    name = filename.replace(".csv", "")

    match = re.match(r"(\d+)_(.+)_vs_NOMINAL", name)

    if match is None:
        raise ValueError(f"Unexpected filename format: {filename}")

    speed = match.group(1)
    fault = match.group(2)

    return speed, fault


# ==========================================================
# Build Mapping
# ==========================================================

mapping = {}

files = sorted(COMPARISON_DIR.glob("*.csv"))

print("=" * 60)
print("BUILDING SENSOR MAPPING")
print("=" * 60)

for file in files:

    speed, fault = parse_filename(file.name)

    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(file)

    # ------------------------------------------------------
    # Remove ignored signals
    # ------------------------------------------------------

    df = df[~df["Sensor"].isin(IGNORE)]

    # ------------------------------------------------------
    # Keep only statistically significant sensors
    # ------------------------------------------------------

    df = df[
        (df["p Value"] <= PVALUE_THRESHOLD) &
        (df["Absolute Effect"] >= MIN_EFFECT)
    ]

    # ------------------------------------------------------
    # Highest ranked sensors
    # ------------------------------------------------------

    df = df.head(TOP_K)

    if df.empty:
        print(f"Warning: No significant sensors found for {fault} ({speed} RPM)")

    sensors = []

    for _, row in df.iterrows():

        sensors.append({

            "sensor": row["Sensor"],

            "importance_score": round(
                float(row["Importance Score"]),
                3
            ),

            "effect_size": round(
                float(row["Absolute Effect"]),
                3
            ),

            "percent_change": round(
                float(row["% Change"]),
                2
            ),

            "p_value": float(
                row["p Value"]
            )

        })

    if fault not in mapping:
        mapping[fault] = {}

    mapping[fault][speed] = {

        "speed": int(speed),

        "fault": fault,

        "top_k": len(sensors),

        "selection": {

            "top_k_requested": TOP_K,

            "p_value_threshold": PVALUE_THRESHOLD,

            "effect_size_threshold": MIN_EFFECT

        },

        "sensors": sensors

    }

    print(f"Selected {len(sensors)} sensors.")


# ==========================================================
# Save Mapping
# ==========================================================

with open(OUTPUT_FILE, "w") as f:
    json.dump(mapping, f, indent=4)

print("\n" + "=" * 60)
print("Sensor mapping saved successfully!")
print(OUTPUT_FILE)
print("=" * 60)