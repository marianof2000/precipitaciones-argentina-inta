import pandas as pd
import pytest

from precipitaciones_argentina.temporal import (
    ACCUMULATED_COLUMN,
    MEAN_COLUMN,
    add_climate_anomalies,
    aggregate_quarterly,
    annual_quarter_values,
    interannual_comparison,
    station_quarter_statistics,
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
        "unidad_original": ["mm"] * 2, "tipo_precipitacion": ["incremental"] * 2,
        "precipitacion_original": [10, 5],
        "precipitacion_mm": [10, 5],
    })
    result = aggregate_quarterly(frame)
    assert result.loc[0, "precipitacion_mm"] == 15
    assert result.loc[0, "cantidad_observaciones"] == 2
    assert result.loc[0, "precipitacion_media_mm"] == 7.5
    assert result.loc[0, "precipitacion_minima_mm"] == 5
    assert result.loc[0, "precipitacion_maxima_mm"] == 10
    assert result.loc[0, "precipitacion_acumulada_mm"] == 15
    assert result.loc[0, ACCUMULATED_COLUMN] == 15
    assert result.loc[0, MEAN_COLUMN] == 7.5


def test_quarterly_rejects_non_accumulated_aggregation():
    frame = pd.DataFrame({
        "dataset_id": ["x"], "archivo_origen": ["x.xls"], "fuente": ["F"],
        "estacion": ["E"], "localidad": ["L"], "provincia": ["P"],
        "latitud": [-34], "longitud": [-58], "anio": [2024],
        "trimestre": ["T1"], "periodo": ["2024-T1"], "unidad_original": ["mm"],
        "tipo_precipitacion": ["incremental"], "precipitacion_original": [10],
        "precipitacion_mm": [10],
    })
    with pytest.raises(ValueError, match="único método admitido"):
        aggregate_quarterly(frame, "mean")


def test_station_quarter_statistics_filters_station_period_and_nan():
    frame = pd.DataFrame({
        "estacion": ["A", "A", "A", "A", "B"],
        "periodo": ["2024-T1", "2024-T1", "2024-T1", "2024-T2", "2024-T1"],
        "precipitacion_mm": [10.0, None, 20.0, 999.0, 888.0],
    })
    assert station_quarter_statistics(frame, "A", "2024-T1") == {
        "mean": 15.0, "sum": 30.0, "min": 10.0, "max": 20.0, "count": 2,
    }


def test_station_quarter_statistics_distinguishes_missing_from_zero():
    frame = pd.DataFrame({
        "estacion": ["A", "B"], "periodo": ["2024-T1", "2024-T1"],
        "precipitacion_mm": [None, 0.0],
    })
    assert station_quarter_statistics(frame, "A", "2024-T1") == {
        "mean": None, "sum": None, "min": None, "max": None, "count": 0,
    }
    assert station_quarter_statistics(frame, "B", "2024-T1") == {
        "mean": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0, "count": 1,
    }


def test_canonical_quarterly_value_is_shared_sum_not_mean():
    values = [0.0, 10.0, 20.0, 25.0, 30.0]
    frame = pd.DataFrame({
        "dataset_id": ["x"] * 5, "archivo_origen": ["x.xls"] * 5,
        "fuente": ["F"] * 5, "estacion": ["Las Armas"] * 5,
        "localidad": ["L"] * 5, "provincia": ["P"] * 5,
        "latitud": [-37.0] * 5, "longitud": [-58.0] * 5,
        "anio": [2015] * 5, "trimestre": ["T3"] * 5,
        "periodo": ["2015-T3"] * 5, "unidad_original": ["mm"] * 5,
        "tipo_precipitacion": ["incremental"] * 5,
        "precipitacion_original": values, "precipitacion_mm": values,
    })
    quarterly = aggregate_quarterly(frame)
    row = quarterly.iloc[0]
    assert row[ACCUMULATED_COLUMN] == 85.0
    assert row[MEAN_COLUMN] == 17.0
    assert row["precipitacion_minima_mm"] == 0.0
    assert row["precipitacion_maxima_mm"] == 30.0
    assert row["cantidad_registros_validos"] == 5
    assert annual_quarter_values(quarterly, "x", "2015-T3")[2]["value"] == 85.0


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


def test_annual_quarter_values_preserves_missing_periods_and_changes_year():
    frame = pd.DataFrame({
        "dataset_id": ["a", "a", "a", "b"],
        "periodo": ["2015-T1", "2015-T3", "2016-T2", "2015-T2"],
        "precipitacion_mm": [120.0, 0.0, 214.0, 999.0],
    })
    result_2015 = annual_quarter_values(frame, "a", "2015-T3")
    assert [item["period"] for item in result_2015] == [
        "2015-T1", "2015-T2", "2015-T3", "2015-T4",
    ]
    assert [item["value"] for item in result_2015] == [120.0, None, 0.0, None]
    result_2016 = annual_quarter_values(frame, "a", "2016-T1")
    assert [item["value"] for item in result_2016] == [None, 214.0, None, None]
