"""Audit all existing evidence outputs to inventory what's available."""
import json, os, pandas as pd

# 1. Profile structure
with open("num_pipeline/outputs/profiles/INCA_SPEED_1000_NOMINAL.json") as f:
    prof = json.load(f)
sensors = list(prof.keys())
print("=== PROFILE STRUCTURE ===")
print(f"Sensors in profile: {len(sensors)}")
print(f"First sensor ({sensors[0]}): {json.dumps(prof[sensors[0]], indent=2)}")
print()

# 2. Comparison file structure
comp = pd.read_csv("num_pipeline/outputs/comparisons/1000_FAULT_INJ_DUR_vs_NOMINAL.csv")
print("=== COMPARISON FILE ===")
print(f"Columns: {list(comp.columns)}")
print(f"Shape: {comp.shape}")
print("First 3 rows:")
print(comp.head(3).to_string())
print()

# 3. EDA report.txt
for root, dirs, files in os.walk("num_pipeline/outputs/eda/INCA_SPEED_1000_NOMINAL"):
    for f in files:
        if f == "report.txt":
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            print(f"=== EDA REPORT ({path}) ===")
            print(content[:2000])
            print()

# 4. Variance ranking head
vr = pd.read_csv("num_pipeline/outputs/eda/INCA_SPEED_1000_NOMINAL/variance_ranking.csv")
print("=== VARIANCE RANKING ===")
print(vr.head(10).to_string())
print()

# 5. Summary statistics
ss = pd.read_csv("num_pipeline/outputs/eda/INCA_SPEED_1000_NOMINAL/summary_statistics.csv")
print("=== SUMMARY STATISTICS ===")
print(ss.head(10).to_string())
print()

# 6. Check for existing PNG files
print("=== EDA PNG FILES ===")
for root, dirs, files in os.walk("num_pipeline/outputs/eda"):
    for f in files:
        if f.endswith(".png"):
            print(os.path.join(root, f))
