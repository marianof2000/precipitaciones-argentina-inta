import pandas as pd

from precipitaciones_argentina.validation import (
    invalid_observation_mask,
    valid_latitude,
    valid_longitude,
)


def test_coordinate_ranges():
    assert valid_latitude(-90) and valid_latitude(90) and not valid_latitude(91)
    assert valid_longitude(-180) and valid_longitude(180) and not valid_longitude(181)


def test_negative_precipitation_is_invalid():
    frame = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01"]), "precipitacion_original": [-1],
        "precipitacion_mm": [-1], "latitud": [-34], "longitud": [-58],
    })
    assert bool(invalid_observation_mask(frame).iloc[0])

