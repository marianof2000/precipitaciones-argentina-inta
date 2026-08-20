import logging

import pandas as pd

from precipitaciones_argentina.preprocessing import normalize_observations


def stations():
    return pd.DataFrame({"id_estacion": ["A1"], "dataset_id": ["A1"], "estacion": ["E"],
        "localidad": ["L"], "provincia": ["P"], "latitud": [-34.0],
        "longitud": [-58.0], "fuente": ["INTA"]})


def test_many_to_one_join_preserves_zero_and_discards_missing():
    raw = pd.DataFrame({"id_estacion": ["A1", "A1"], "fecha": ["2024-01-01", "2024-01-02"],
                        "precipitacion_pluviometrica": [0.0, None]})
    result, metrics = normalize_observations(raw, stations())
    assert result["precipitacion_mm"].tolist() == [0.0]
    assert metrics["missing"] == 1


def test_unknown_station_is_warned_and_omitted(caplog):
    raw = pd.DataFrame({"id_estacion": ["UNKNOWN"], "fecha": ["2024-01-01"],
                        "precipitacion_pluviometrica": [2.0]})
    with caplog.at_level(logging.WARNING):
        result, metrics = normalize_observations(raw, stations())
    assert result.empty
    assert metrics["unknown_station_ids"] == ["UNKNOWN"]
    assert "no están en el catálogo" in caplog.text
