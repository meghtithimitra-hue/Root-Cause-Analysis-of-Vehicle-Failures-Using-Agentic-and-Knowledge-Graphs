import json
from pathlib import Path


def load_current_sensor_sample(speed):

    """
    Temporary implementation.

    Until live vehicle data is available,
    we use the statistical mean of the
    nominal profile as the current sensor
    readings.
    """

    profile = Path(
        f"outputs/profiles/INCA_SPEED_{speed}_NOMINAL.json"
    )

    with open(profile) as f:

        data = json.load(f)

    sample = {}

    for sensor, stats in data.items():

        if isinstance(stats, dict):

            sample[sensor] = stats["mean"]

    return sample