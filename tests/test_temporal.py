import pandas as pd

from precipitaciones_argentina.temporal import aggregate_quarterly, year_quarter


def test_year_quarter_and_period():
    assert year_quarter(pd.Timestamp("2024-05-20")) == (2024, "T2", "2024-T2")


def test_quarterly_sum():
    frame = pd.DataFrame({
        "dataset_id": ["x", "x"], "archivo_origen": ["x.xls"] * 2,
        "fuente": ["F"] * 2, "estacion": ["E"] * 2, "localidad": ["L"] * 2,
        "provincia": ["P"] * 2, "latitud": [-34] * 2, "longitud": [-58] * 2,
        "anio": [2024] * 2, "trimestre": ["T1"] * 2, "periodo": ["2024-T1"] * 2,
        "unidad_original": ["mm"] * 2, "precipitacion_original": [10, 5],
        "precipitacion_mm": [10, 5],
    })
    result = aggregate_quarterly(frame)
    assert result.loc[0, "precipitacion_mm"] == 15
    assert result.loc[0, "cantidad_observaciones"] == 2

