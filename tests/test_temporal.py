import pandas as pd

from precipitaciones_argentina.temporal import (
    add_climate_anomalies,
    aggregate_quarterly,
    interannual_comparison,
    station_time_series,
    year_quarter,
)


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


def test_climate_anomalies_and_zero_baseline():
    frame = pd.DataFrame({
        "dataset_id": ["a", "a", "a", "b"],
        "trimestre": ["T1"] * 4,
        "anio": [1991, 1992, 2024, 1991],
        "precipitacion_mm": [100.0, 200.0, 210.0, 0.0],
    })
    result = add_climate_anomalies(frame, minimum_years=1)
    current = result.loc[result["anio"].eq(2024)].iloc[0]
    assert current["precipitacion_historica_mm"] == 150
    assert current["anomalia_absoluta_mm"] == 60
    assert current["anomalia_relativa_pct"] == 40
    assert pd.isna(result.loc[result["dataset_id"].eq("b"), "anomalia_relativa_pct"]).all()


def test_station_series_and_interannual_comparison():
    frame = pd.DataFrame({
        "dataset_id": ["a", "a", "a"], "anio": [2024, 2023, 2024],
        "trimestre": ["T2", "T1", "T1"], "precipitacion_mm": [1, 2, 3],
    })
    assert station_time_series(frame, "a")["precipitacion_mm"].tolist() == [2, 3, 1]
    assert interannual_comparison(frame, "a", "T1")["anio"].tolist() == [2023, 2024]
