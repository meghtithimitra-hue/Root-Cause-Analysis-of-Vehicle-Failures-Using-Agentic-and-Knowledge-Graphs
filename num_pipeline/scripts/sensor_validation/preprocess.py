from pathlib import Path

from load_data import NavicLoader


RAW_FOLDER = Path("data/raw")

OUTPUT_FOLDER = Path("data/processed")

def preprocess(df):

    # remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]

    # remove empty columns
    df = df.dropna(axis=1, how="all")

    # interpolate missing values
    df = df.interpolate()

    # remove columns with one unique value
    nunique = df.nunique()

    constant = nunique[nunique <= 1].index

    df = df.drop(columns=constant)

    return df

for file in RAW_FOLDER.glob("*.xlsx"):

    loader = NavicLoader(file)

    df, units = loader.load()

    df = preprocess(df)

    out = OUTPUT_FOLDER / (file.stem + ".csv")

    df.to_csv(out, index=False)

    print(file.name, "done")