from pathlib import Path

import numpy as np
import pandas as pd

from precipitaciones_argentina.visualization import (
    anomaly_rgba,
    build_compact_temporal_payload,
    generate_map,
    precipitation_rgba,
)


def sample_frame():
    return pd.DataFrame({
        "dataset_id": ["a", "a"], "archivo_origen": ["a.xls", "a.xls"],
        "fuente": ["F", "F"], "estacion": ["E", "E"], "localidad": ["L", "L"],
        "provincia": ["P", "P"], "latitud": [-34.0, -34.0],
        "longitud": [-58.0, -58.0], "anio": [2023, 2024],
        "trimestre": ["T4", "T1"], "periodo": ["2023-T4", "2024-T1"],
        "precipitacion_original": [20.0, 30.0], "unidad_original": ["mm", "mm"],
        "precipitacion_mm": [20.0, 30.0], "cantidad_observaciones": [3, 4],
    })


def test_payload_is_chronological():
    payload = build_compact_temporal_payload(sample_frame().iloc[::-1])
    assert list(payload["periods"]) == ["2023-T4", "2024-T1"]
    assert len(payload["stations"]) == 1


def test_precipitation_raster_is_transparent_outside_mask():
    rgba = precipitation_rgba(
        np.array([[0.0, 50.0]]), np.array([[False, True]]), maximum=100
    )
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] > 0


def test_same_precipitation_always_has_same_color():
    values = np.array([[137.0], [137.0]])
    rgba = precipitation_rgba(values, np.ones_like(values, dtype=bool), maximum=500)
    assert np.array_equal(rgba[0], rgba[1])


def test_anomaly_scale_is_diverging_and_transparent_for_missing_cells():
    values = np.array([[-100.0, 0.0, 100.0, np.nan]])
    rgba = anomaly_rgba(values, np.array([[True, True, True, False]]), limit=100)
    assert rgba[0, 0, 2] == 255
    assert rgba[0, 1, :3].tolist() == [255, 255, 255]
    assert rgba[0, 2, 0] == 255
    assert rgba[0, 3, 3] == 0


def test_generate_static_map(tmp_path):
    provinces = Path(__file__).parents[1] / "assets" / "argentina_provincias.geojson"
    output = tmp_path / "index.html"
    generate_map(sample_frame(), provinces, output)
    html = output.read_text(encoding="utf-8")
    assert "period-slider" in html
    assert "2024-T1" in html
    assert "Observaciones reales" in html
