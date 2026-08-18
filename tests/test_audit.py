import pandas as pd

from precipitaciones_argentina.audit import build_traceability_samples


def test_traceability_sample_matches_quarterly_value():
    daily = pd.DataFrame({
        "dataset_id": ["a", "a"], "archivo_origen": ["datos/a.xls"] * 2,
        "fuente": ["F"] * 2, "estacion": ["E"] * 2, "localidad": ["L"] * 2,
        "provincia": ["P"] * 2, "latitud": [-34.0] * 2, "longitud": [-58.0] * 2,
        "fecha": pd.to_datetime(["2024-01-01", "2024-02-01"]), "anio": [2024] * 2,
        "trimestre": ["T1"] * 2, "periodo": ["2024-T1"] * 2,
        "precipitacion_original": [10.0, 20.0], "unidad_original": ["mm"] * 2,
        "precipitacion_mm": [10.0, 20.0],
    })
    quarterly = pd.DataFrame({
        "dataset_id": ["a"], "periodo": ["2024-T1"], "precipitacion_mm": [30.0]
    })
    sample = build_traceability_samples(daily, quarterly, sample_count=1)[0]
    assert sample["coincide_agregacion"]
    assert sample["acumulado_parquet_mapa_mm"] == 30
