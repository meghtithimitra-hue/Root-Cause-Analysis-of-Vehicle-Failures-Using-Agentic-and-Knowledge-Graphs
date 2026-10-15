from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Generate Summary Statistics
# -----------------------------

def generate_summary(df):

    summary = pd.DataFrame(index=df.columns)

    summary["Data Type"] = df.dtypes.astype(str)
    summary["Count"] = df.count()
    summary["Missing"] = df.isnull().sum()
    summary["Unique"] = df.nunique()

    summary["Mean"] = df.mean()
    summary["Median"] = df.median()
    summary["Std"] = df.std()
    summary["Variance"] = df.var()

    summary["Min"] = df.min()
    summary["25%"] = df.quantile(0.25)
    summary["75%"] = df.quantile(0.75)
    summary["Max"] = df.max()

    return summary


# -----------------------------
# Dataset Report
# -----------------------------

def save_report(df, folder):

    report = []

    report.append(f"Rows               : {df.shape[0]}")
    report.append(f"Columns            : {df.shape[1]}")
    report.append(f"Missing Values     : {df.isnull().sum().sum()}")
    report.append(f"Duplicate Rows     : {df.duplicated().sum()}")

    numeric_cols = df.select_dtypes(include="number").shape[1]

    report.append(f"Numeric Columns    : {numeric_cols}")

    with open(folder / "report.txt", "w") as f:

        for line in report:
            f.write(line + "\n")


# -----------------------------
# Correlation Matrix
# -----------------------------

def save_correlation(df, folder):

    corr = df.corr(numeric_only=True)

    corr.to_csv(folder / "correlation_matrix.csv")

    plt.figure(figsize=(12,10))

    plt.imshow(corr)

    plt.colorbar()

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(folder / "correlation_matrix.png")

    plt.close()


# -----------------------------
# Histograms
# -----------------------------

def save_histograms(df, folder):

    hist_folder = folder / "histograms"

    hist_folder.mkdir(exist_ok=True)

    for column in df.columns:

        plt.figure(figsize=(6,4))

        plt.hist(df[column].dropna(), bins=40)

        plt.title(column)

        plt.xlabel(column)

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(hist_folder / f"{column}.png")

        plt.close()


# -----------------------------
# Boxplots
# -----------------------------

def save_boxplots(df, folder):

    box_folder = folder / "boxplots"

    box_folder.mkdir(exist_ok=True)

    for column in df.columns:

        plt.figure(figsize=(5,5))

        plt.boxplot(df[column].dropna())

        plt.title(column)

        plt.tight_layout()

        plt.savefig(box_folder / f"{column}.png")

        plt.close()


# -----------------------------
# Variance Ranking
# -----------------------------

def save_variance(df, folder):

    variance = pd.DataFrame()

    variance["Variance"] = df.var()

    variance = variance.sort_values(
        by="Variance",
        ascending=False
    )

    variance.to_csv(folder / "variance_ranking.csv")


# -----------------------------
# Main Function
# -----------------------------

def run_eda(csv_file):

    print(f"\nProcessing : {csv_file.name}")

    df = pd.read_csv(csv_file)

    output_folder = OUTPUT_DIR / csv_file.stem

    output_folder.mkdir(parents=True, exist_ok=True)

    summary = generate_summary(df)

    summary.to_csv(
        output_folder / "summary_statistics.csv"
    )

    save_report(df, output_folder)

    save_correlation(df, output_folder)

    save_variance(df, output_folder)

    save_histograms(df, output_folder)

    save_boxplots(df, output_folder)

    print("Completed")


# -----------------------------
# Run on all CSV files
# -----------------------------

if __name__ == "__main__":

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    for file in csv_files:

        run_eda(file)

    print("\n===================================")
    print("EDA COMPLETED FOR ALL DATASETS")
    print("===================================")