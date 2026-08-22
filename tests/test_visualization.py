import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box

from precipitaciones_argentina import config
from precipitaciones_argentina.spatial import create_spatial_grid
from precipitaciones_argentina.visualization import (
    anomaly_rgba,
    build_compact_temporal_payload,
    build_interpolation_payload,
    generate_map,
    orient_rgba_for_leaflet,
    precipitation_rgba,
    precipitation_to_rgba,
)


def sample_frame():
    return pd.DataFrame({
        "dataset_id": ["a", "a"], "archivo_origen": ["estaciones.csv", "estaciones.csv"],
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
    assert rgba[0, 1, 3] == 255
    zero = precipitation_rgba(
        np.array([[0.0]]), np.array([[True]]), maximum=100
    )
    assert zero[0, 0, 3] > 0


def test_leaflet_raster_has_north_up_and_west_left():
    south_west = [255, 0, 0, 175]
    south_east = [0, 255, 0, 175]
    north_west = [0, 0, 255, 175]
    north_east = [255, 255, 0, 175]
    rgba = np.array([[south_west, south_east], [north_west, north_east]], dtype=np.uint8)
    oriented = orient_rgba_for_leaflet(rgba)
    assert oriented[0, 0].tolist() == north_west
    assert oriented[0, 1].tolist() == north_east
    assert oriented[1, 0].tolist() == south_west


def test_same_precipitation_always_has_same_color():
    values = np.array([[137.0], [137.0]])
    rgba = precipitation_rgba(values, np.ones_like(values, dtype=bool), maximum=500)
    assert np.array_equal(rgba[0], rgba[1])


def test_marker_and_raster_share_logarithmic_color():
    value, maximum = 85.0, 1260.5
    marker_rgba = precipitation_to_rgba(
        value, maximum_mm=maximum, scale="log", alpha=1
    )
    raster_rgba = precipitation_rgba(
        np.array([[value]]), np.array([[True]]), maximum, scale="log"
    )
    assert marker_rgba == tuple(raster_rgba[0, 0])


def test_legend_ticks_and_raster_share_color_function():
    maximum = 1260.5
    for value in [10.0, 50.0, 100.0, 200.0, 500.0]:
        legend_rgba = precipitation_to_rgba(
            value, maximum_mm=maximum, scale="log"
        )
        raster_rgba = precipitation_rgba(
            np.array([[value]]), np.array([[True]]), maximum, scale="log"
        )
        assert legend_rgba == tuple(raster_rgba[0, 0])


def test_python_and_javascript_color_calculation_are_equivalent():
    node = shutil.which("node")
    if node is None:
        return
    values = [0, 0.5, 10, 25, 50, 85, 100, 200, 500, 1260.5]
    script = f"""
const stops=[
 [0,'#ffffd9'],[.2,'#c7e9b4'],[.4,'#7fcdbb'],
 [.6,'#41b6c4'],[.8,'#225ea8'],[1,'#081d58']
];
const maximum=1260.5;
function color(value,scale){{
 const bounded=Math.max(0,Math.min(maximum,value));
 const ratio=scale==='log'?Math.log1p(bounded)/Math.log1p(maximum):bounded/maximum;
 const rgb=hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
 for(let i=0;i<stops.length-1;i++){{const a=stops[i],b=stops[i+1];if(ratio<=b[0]){{
  const f=(ratio-a[0])/(b[0]-a[0]),x=rgb(a[1]),y=rgb(b[1]);
  return x.map((v,j)=>Math.round(v+f*(y[j]-v))).concat(255);
 }}}}
 return rgb(stops.at(-1)[1]).concat(255);
}}
console.log(JSON.stringify({json.dumps(values)}.flatMap(v=>['linear','log'].map(s=>[v,s,color(v,s)]))));
"""
    javascript = json.loads(
        subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout
    )
    for value, scale, rgba in javascript:
        assert precipitation_to_rgba(
            value, maximum_mm=1260.5, scale=scale
        ) == tuple(rgba)


def test_idw_includes_active_station_with_unassigned_province():
    frame = pd.DataFrame({
        "periodo": ["2024-T1"] * 3,
        "dataset_id": ["a", "b", "chascomus"],
        "estacion": ["A", "B", "Chascomus - EEA Cuenca Salado"],
        "provincia": ["P", "P", "Sin asignar"],
        "latitud": [-35.0, -35.0, -35.5],
        "longitud": [-58.5, -57.5, -58.0],
        "precipitacion_mm": [100.0, 200.0, 184.2],
        "anomalia_absoluta_mm": [1.0, 2.0, 3.0],
        "anomalia_relativa_pct": [1.0, 2.0, 3.0],
    })
    grid = create_spatial_grid(box(-59, -36, -57, -34), 0.5)
    payload = build_interpolation_payload(frame, grid, 200, 3, 200)
    assert payload["2024-T1"]["absolute"]["station_count"] == 3
    assert set(payload["2024-T1"]["absolute"]["images"]) == {"linear", "log"}
    assert "image" not in payload["2024-T1"]["absolute"]
    assert (
        payload["2024-T1"]["absolute"]["images"]["linear"]
        != payload["2024-T1"]["absolute"]["images"]["log"]
    )


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
    assert "Promedio de los registros" in html
    assert "Acumulado trimestral" in html
    assert "Sin registros de precipitación disponibles" in html
    assert "Exclusivamente observaciones reales" in html
    assert "color-scale-select" in html
    assert "Logarítmica" in html
    assert "Math.log1p" in html
    assert "Trimestres del año seleccionado" in html
    assert "updateAnnualQuarterChart" in html
    assert "Precipitación acumulada trimestral — Año:" in html
    assert "chartGrid" in html
    assert "stroke=\"#d7dce0\"" in html
    assert "#legend-panel[open]" in html
    assert '<details id="legend-panel" class="precip-panel" open>' in html
    assert "localeCompare(b.station.e,'es'" in html
    assert "idw-opaque-checkbox" in html
    assert "let idwOpaque = idwOpaqueCheckbox.checked" in html
    assert "estimation.images[colorScaleSelect.value]" in html
    assert "interpolationOverlay.setOpacity" in html
    assert "const idwDefaultOpacity = 0.45" in html
    assert "const hasFocus=query.has('lat') && query.has('lon')" in html
    assert "if(hasFocus && Number.isFinite(focusLat)" in html
    assert f"center: [{config.ARGENTINA_CENTER[0]}, {config.ARGENTINA_CENTER[1]}]" in html
