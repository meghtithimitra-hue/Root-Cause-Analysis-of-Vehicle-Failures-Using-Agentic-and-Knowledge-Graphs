from pathlib import Path
import pandas as pd
import json

# ==========================================================
# Paths
# ==========================================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/profiles")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Build statistical profile
# ==========================================================

def build_profile(df):

    profile = {}

    for column in df.columns:

        # Skip non-numeric columns (if any)
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        profile[column] = {

            "count": int(df[column].count()),

            "mean": round(float(df[column].mean()), 6),

            "std": round(float(df[column].std()), 6),

            "variance": round(float(df[column].var()), 6),

            "min": round(float(df[column].min()), 6),

            "max": round(float(df[column].max()), 6),

            "median": round(float(df[column].median()), 6),

            "q1": round(float(df[column].quantile(0.25)), 6),

            "q3": round(float(df[column].quantile(0.75)), 6),

            "iqr": round(
                float(
                    df[column].quantile(0.75)
                    -
                    df[column].quantile(0.25)
                ),
                6
            ),

            "skewness": round(float(df[column].skew()), 6),

            "kurtosis": round(float(df[column].kurt()), 6)
        }

    return profile


# ==========================================================
# Save profile
# ==========================================================

def save_profile(profile, output_file):

    with open(output_file, "w") as f:

        json.dump(profile, f, indent=4)


# ==========================================================
# Process one dataset
# ==========================================================

def process_file(file):

    print(f"Building profile : {file.name}")

    df = pd.read_csv(file)

    profile = build_profile(df)

    output_file = OUTPUT_DIR / (file.stem + ".json")

    save_profile(profile, output_file)

    print(f"Saved : {output_file}\n")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print("=" * 60)
    print("BUILDING STATISTICAL PROFILES")
    print("=" * 60)

    for file in csv_files:

        process_file(file)

    print("=" * 60)
    print("ALL PROFILES GENERATED")
    print("=" * 60)