import pandas as pd
import pytest

from precipitaciones_argentina.catalog import DatasetConfig
from precipitaciones_argentina.preprocessing import normalize_dataset, precipitation_to_mm


@pytest.mark.parametrize(("unit", "expected"), [("mm", 2), ("cm", 20), ("in", 50.8)])
def test_precipitation_conversion(unit, expected):
    assert precipitation_to_mm(pd.Series([2]), unit).iloc[0] == expected


def test_duplicate_detection_and_missing_not_zero():
    raw = pd.DataFrame({
        "Fecha": ["2024-01-01", "2024-01-01", "bad"], "Lluvia": [2, 3, None]
    })
    cfg = DatasetConfig(
        "x", "x.xls", "F", "E", "L", "P", -34, -58, 0, "mm",
        {"fecha": "Fecha", "precipitacion": "Lluvia"},
    )
    result, metrics = normalize_dataset(raw, cfg)
    assert len(result) == 1
    assert metrics["duplicates"] == 1
    assert metrics["missing"] == 1

