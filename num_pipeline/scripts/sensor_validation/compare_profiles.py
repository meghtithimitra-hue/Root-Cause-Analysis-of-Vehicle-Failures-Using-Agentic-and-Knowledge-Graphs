from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ==========================================================
# Paths
# ==========================================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/comparisons")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Cohen's d
# ==========================================================

def cohens_d(x, y):

    x = x.dropna()
    y = y.dropna()

    nx = len(x)
    ny = len(y)

    pooled_std = np.sqrt(
        ((nx - 1) * x.var() + (ny - 1) * y.var())
        / (nx + ny - 2)
    )

    if pooled_std == 0:
        return 0

    return (x.mean() - y.mean()) / pooled_std


# ==========================================================
# Compare two datasets
# ==========================================================

def compare_datasets(nominal_df, fault_df):

    results = []

    common_columns = nominal_df.columns.intersection(
        fault_df.columns
    )

    for col in common_columns:

        x = nominal_df[col]
        y = fault_df[col]

        mean_nominal = x.mean()
        mean_fault = y.mean()

        std_nominal = x.std()
        std_fault = y.std()

        # ---------------------------------------
        # Percentage Change
        # ---------------------------------------

        if mean_nominal != 0:
            pct_change = (
                (mean_fault - mean_nominal)
                / abs(mean_nominal)
            ) * 100
        else:
            pct_change = np.nan

        # ---------------------------------------
        # Welch's t-test
        # ---------------------------------------

        t_stat, p_value = ttest_ind(
            x,
            y,
            equal_var=False,
            nan_policy="omit"
        )

        # ---------------------------------------
        # Effect Size
        # ---------------------------------------

        effect = cohens_d(x, y)

        results.append({

            "Sensor": col,

            "Nominal Mean": mean_nominal,

            "Fault Mean": mean_fault,

            "Nominal Std": std_nominal,

            "Fault Std": std_fault,

            "% Change": pct_change,

            "t Statistic": t_stat,

            "p Value": p_value,

            "Effect Size": effect,

            "Absolute Effect": abs(effect)

        })

    result_df = pd.DataFrame(results)

    # =====================================================
    # Composite Feature Importance
    # =====================================================

    # Normalize Absolute Effect
    max_effect = result_df["Absolute Effect"].max()

    if max_effect != 0:
        result_df["Effect_Norm"] = (
            result_df["Absolute Effect"] / max_effect
        )
    else:
        result_df["Effect_Norm"] = 0

    # Normalize Percentage Change
    max_pct = result_df["% Change"].abs().max()

    if max_pct != 0:
        result_df["Pct_Norm"] = (
            result_df["% Change"].abs() / max_pct
        )
    else:
        result_df["Pct_Norm"] = 0

    # Normalize p-value
    result_df["P_Norm"] = -np.log10(
        result_df["p Value"] + 1e-12
    )

    max_p = result_df["P_Norm"].max()

    if max_p != 0:
        result_df["P_Norm"] = (
            result_df["P_Norm"] / max_p
        )
    else:
        result_df["P_Norm"] = 0

    # =====================================================
    # Importance Score
    # =====================================================

    result_df["Importance Score"] = (

        0.50 * result_df["Effect_Norm"]

        +

        0.30 * result_df["Pct_Norm"]

        +

        0.20 * result_df["P_Norm"]

    )

    # =====================================================
    # Ranking
    # =====================================================

    result_df = result_df.sort_values(

        by="Importance Score",

        ascending=False

    )

    result_df.reset_index(

        drop=True,

        inplace=True

    )

    result_df.insert(

        0,

        "Rank",

        result_df.index + 1

    )

    return result_df


# ==========================================================
# Find Matching Files
# ==========================================================

def get_file(speed, condition):

    filename = f"INCA_SPEED_{speed}_{condition}.csv"

    return DATA_DIR / filename


# ==========================================================
# Compare One Speed
# ==========================================================

def compare_speed(speed):

    nominal = pd.read_csv(
        get_file(speed, "NOMINAL")
    )

    fault_types = [

        "FAULT_INJ_DUR",

        "FAULT_INJ_PRS",

        "FAULT_SOI"

    ]

    for fault in fault_types:

        print(f"\n{speed} RPM : {fault}")

        fault_df = pd.read_csv(

            get_file(speed, fault)

        )

        comparison = compare_datasets(

            nominal,

            fault_df

        )

        out_file = (

            OUTPUT_DIR /

            f"{speed}_{fault}_vs_NOMINAL.csv"

        )

        comparison.to_csv(

            out_file,

            index=False

        )

        print("Saved:", out_file)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERATING COMPARISON REPORTS")
    print("=" * 60)

    speeds = [

        1000,

        1200,

        1400,

        1600

    ]

    for speed in speeds:

        compare_speed(speed)

    print("\n" + "=" * 60)
    print("ALL COMPARISONS COMPLETED")
    print("=" * 60)