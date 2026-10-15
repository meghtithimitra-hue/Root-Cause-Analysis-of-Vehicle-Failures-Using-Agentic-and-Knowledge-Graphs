# scripts/sensor_validation/load_data.py

from pathlib import Path
import pandas as pd


class NavicLoader:

    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def load(self):

        # first row = headers
        headers = pd.read_excel(
            self.filepath,
            header=None,
            nrows=1
        ).iloc[0]

        # second row = engineering units
        units = pd.read_excel(
            self.filepath,
            header=None,
            skiprows=1,
            nrows=1
        ).iloc[0]

        # actual sensor data
        df = pd.read_excel(
            self.filepath,
            header=None,
            skiprows=2
        )

        df.columns = headers

        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        unit_dict = dict(zip(headers, units))

        return df, unit_dict