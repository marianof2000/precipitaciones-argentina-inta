# ruff: noqa: E501
"""Generación del mapa temporal estático con Folium y Leaflet.

Las líneas extensas corresponden a JavaScript y CSS embebidos en el HTML estático.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import Element, MacroElement, Template
from folium.plugins import HeatMap
from PIL import Image

from . import config
from .spatial import SpatialGrid, create_spatial_grid, idw_interpolation, load_territory
from .statistics import (
    generate_precipitation_ticks,
    precipitation_global_maximum,
    quarterly_statistics,
)

COLOR_STOPS = [
    (0.0, "#ffffd9"),
    (0.2, "#c7e9b4"),
    (0.4, "#7fcdbb"),
    (0.6, "#41b6c4"),
    (0.8, "#225ea8"),
    (1.0, "#081d58"),
]


class _DeferredScript(MacroElement):
    """Inserta JavaScript después de que Folium haya creado mapa y capas."""

    def __init__(self, script: str) -> None:
        super().__init__()
        self._template = Template(f"{{% macro script(this, kwargs) %}}{script}{{% endmacro %}}")


def _period_key(period: str) -> tuple[int, int]:
    year, quarter = period.split("-T")
    return int(year), int(quarter)


def build_compact_temporal_payload(frame: pd.DataFrame) -> dict[str, object]:
    """Deduplica metadatos de estaciones para reducir el HTML sin perder campos."""
    station_fields = [
        "dataset_id", "archivo_origen", "fuente", "estacion", "localidad", "provincia",
        "latitud", "longitud", "unidad_original",
    ]
    stations = frame.drop_duplicates("dataset_id")[station_fields].reset_index(drop=True)
    station_index = {dataset_id: index for index, dataset_id in enumerate(stations["dataset_id"])}
    station_records = [
        {
            "d": row.dataset_id,
            "a": row.archivo_origen,
            "f": row.fuente,
            "e": row.estacion,
            "l": row.localidad,
            "p": row.provincia,
            "y": round(float(row.latitud), 6),
            "x": round(float(row.longitud), 6),
            "u": row.unidad_original,
        }
        for row in stations.itertuples(index=False)
    ]
    periods: dict[str, list[list[float | int]]] = {}
    for period, rows in frame.groupby("periodo", sort=False):
        periods[str(period)] = [
            [
                station_index[row.dataset_id],
                round(float(row.precipitacion_original), 6),
                round(float(row.precipitacion_mm), 6),
                int(row.cantidad_observaciones),
            ]
            for row in rows.itertuples(index=False)
        ]
    return {
        "stations": station_records,
        "periods": dict(sorted(periods.items(), key=lambda item: _period_key(item[0]))),
    }


def _statistics_payload(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    stats = quarterly_statistics(frame)
    payload: dict[str, dict[str, float | int]] = {}
    for row in stats.to_dict(orient="records"):
        period = str(row.pop("periodo"))
        payload[period] = {
            key: int(value) if key.startswith("cantidad_") else round(float(value), 2)
            for key, value in row.items()
        }
    return payload


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def precipitation_rgba(
    values: np.ndarray, valid_mask: np.ndarray, maximum: float, alpha: int = 175
) -> np.ndarray:
    """Aplica la misma escala continua global del mapa y transparencia fuera de cobertura."""
    ratio = np.clip(values / maximum, 0, 1) if maximum > 0 else np.zeros_like(values)
    result = np.zeros((*values.shape, 4), dtype=np.uint8)
    for (start, start_color), (end, end_color) in zip(
        COLOR_STOPS, COLOR_STOPS[1:], strict=False
    ):
        selection = valid_mask & (ratio >= start) & (ratio <= end)
        fraction = np.divide(
            ratio - start,
            end - start,
            out=np.zeros_like(values, dtype=float),
            where=end != start,
        )
        first = np.asarray(_hex_to_rgb(start_color))
        last = np.asarray(_hex_to_rgb(end_color))
        colors = first + fraction[..., None] * (last - first)
        result[selection, :3] = colors[selection].astype(np.uint8)
    result[valid_mask, 3] = alpha
    return result


def _rgba_data_url(rgba: np.ndarray) -> str:
    """Codifica un ráster RGBA como PNG embebible."""
    buffer = io.BytesIO()
    Image.fromarray(np.flipud(rgba), mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_interpolation_payload(
    frame: pd.DataFrame, grid: SpatialGrid, maximum: float
) -> dict[str, dict[str, object]]:
    """Precalcula una superficie IDW transparente por período."""
    payload: dict[str, dict[str, object]] = {}
    for period, rows in frame.groupby("periodo", sort=False):
        rows = rows.loc[rows["provincia"].str.casefold().ne("sin asignar")]
        result = idw_interpolation(
            rows["longitud"].to_numpy(dtype=float),
            rows["latitud"].to_numpy(dtype=float),
            rows["precipitacion_mm"].to_numpy(dtype=float),
            grid,
            power=config.IDW_POWER,
            maximum_distance_km=config.MAX_INTERPOLATION_DISTANCE_KM,
            minimum_stations=config.MIN_INTERPOLATION_STATIONS,
        )
        url = None
        if result.valid_mask.any():
            url = _rgba_data_url(
                precipitation_rgba(result.values, result.valid_mask, maximum)
            )
        payload[str(period)] = {
            "image": url,
            "station_count": result.station_count,
            "has_estimation": url is not None,
        }
    return dict(sorted(payload.items(), key=lambda item: _period_key(item[0])))


def _controls_html(maximum: float) -> str:
    stops = ", ".join(color for _, color in COLOR_STOPS)
    return f"""
    <style>
      .precip-panel {{ position: fixed; z-index: 1000; background: rgba(255,255,255,.96);
        box-shadow: 0 1px 7px rgba(0,0,0,.35); border-radius: 6px; font: 13px/1.4 Arial;
        color: #17202a; }}
      #time-panel {{ left: 50%; bottom: 24px; transform: translateX(-50%); width: min(680px,82vw);
        padding: 10px 14px; }}
      #time-row {{ display:flex; align-items:center; gap:10px; }}
      #period-slider {{ flex:1; }} #period-label {{ font-size:17px; font-weight:700; min-width:78px; }}
      #play-period {{ border:1px solid #777; background:#fff; border-radius:4px; cursor:pointer; }}
      #stats-panel {{ right: 12px; top: 150px; width: 245px; padding: 12px; }}
      #stats-panel h4 {{ margin:0 0 7px; font-size:15px; }}
      #stats-grid {{ display:grid; grid-template-columns:1fr auto; gap:3px 10px; }}
      #legend-panel {{ left: 12px; bottom: 24px; width: 185px; padding:10px; }}
      #legend-gradient {{ height:12px; background:linear-gradient(to right,{stops}); margin:6px 0 2px; }}
      #legend-values {{ display:flex; justify-content:space-between; }}
      .leaflet-popup-content table {{ border-collapse:collapse; }}
      .leaflet-popup-content td {{ padding:2px 5px; border-bottom:1px solid #eee; }}
      @media(max-width:700px) {{ #stats-panel {{ top:180px; width:190px; }}
        #legend-panel {{ display:none; }} #time-panel {{ width:88vw; }} }}
    </style>
    <div id="time-panel" class="precip-panel">
      <div id="time-row"><button id="play-period" title="Reproducir">▶</button>
      <button id="previous-period" title="Período anterior">◀</button>
      <input id="period-slider" type="range" min="0" value="0" step="1">
      <button id="next-period" title="Período siguiente">▶</button>
      <span id="period-label">—</span></div>
      <div>Precipitación acumulada trimestral · flechas ←/→ para navegar</div>
    </div>
    <div id="stats-panel" class="precip-panel"><h4>Estadísticas del período</h4>
      <div id="stats-grid"></div></div>
    <div id="legend-panel" class="precip-panel"><strong>Precipitación observada (mm)</strong>
      <div id="legend-gradient"></div><div id="legend-values"><span>0</span>
      <span>{maximum:g}</span></div><small>Escala global · cortes cada 10 mm</small><br>
      <small>● observado &nbsp; ≠ estimación espacial</small></div>
    """


def _map_script(
    observations_name: str,
    interpolation_name: str,
    payload: dict[str, object],
    statistics: dict[str, dict[str, float | int]],
    interpolation_payload: dict[str, dict[str, object]],
    interpolation_bounds: list[list[float]],
    maximum: float,
) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    stats_json = json.dumps(statistics, ensure_ascii=False, separators=(",", ":"))
    interpolation_json = json.dumps(
        interpolation_payload, ensure_ascii=False, separators=(",", ":")
    )
    bounds_json = json.dumps(interpolation_bounds)
    colors_json = json.dumps(COLOR_STOPS)
    return f"""
    const precipitationData = {data_json};
    const periodStatistics = {stats_json};
    const interpolationData = {interpolation_json};
    const precipitationMaximum = {maximum};
    const precipitationColors = {colors_json};
    const stationMetadata = precipitationData.stations;
    const periodRows = precipitationData.periods;
    const periods = Object.keys(periodRows);
    const slider = document.getElementById('period-slider');
    const label = document.getElementById('period-label');
    const statsGrid = document.getElementById('stats-grid');
    const playButton = document.getElementById('play-period');
    const previousButton = document.getElementById('previous-period');
    const nextButton = document.getElementById('next-period');
    slider.max = Math.max(0, periods.length - 1);
    slider.value = Math.max(0, periods.length - 1);
    let timer = null;
    let interpolationOverlay = null;

    function escapeHtml(value) {{
      return String(value ?? '—').replace(/[&<>'"]/g, char =>
        ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}})[char]);
    }}
    function colorFor(value) {{
      const ratio = precipitationMaximum > 0 ? Math.max(0, Math.min(1, value / precipitationMaximum)) : 0;
      function rgb(hex) {{ return [1,3,5].map(i => parseInt(hex.slice(i,i+2),16)); }}
      for (let i=0; i<precipitationColors.length-1; i++) {{
        const first=precipitationColors[i], last=precipitationColors[i+1];
        if (ratio <= last[0]) {{
          const f=(ratio-first[0])/(last[0]-first[0]); const a=rgb(first[1]), b=rgb(last[1]);
          return `rgb(${{a.map((v,j) => Math.round(v+f*(b[j]-v))).join(',')}})`;
        }}
      }}
      return precipitationColors[precipitationColors.length-1][1];
    }}
    function popup(station,row,period) {{
      const [year,quarter] = period.split('-');
      const fields = [['Estación',station.e],['Localidad',station.l],
        ['Provincia',station.p],['Fuente',station.f],['Dataset',station.d],
        ['Archivo',station.a],['Año',year],['Trimestre',quarter],['Período',period],
        ['Precipitación original',`${{row[1]}} ${{station.u}}`],
        ['Precipitación en mm',`${{Number(row[2]).toFixed(1)}} mm`],
        ['Latitud',station.y],['Longitud',station.x],['Tipo','Dato observado']];
      return '<table>' + fields.map(item => `<tr><td><b>${{escapeHtml(item[0])}}</b></td><td>${{escapeHtml(item[1])}}</td></tr>`).join('') + '</table>';
    }}
    function renderPeriod(index) {{
      const period = periods[index]; if (!period) return;
      {observations_name}.clearLayers();
      for (const row of periodRows[period]) {{
        const station = stationMetadata[row[0]]; const color = colorFor(Number(row[2]));
        L.circleMarker([station.y,station.x], {{radius:6,color:'#263238',weight:.6,
          fillColor:color,fillOpacity:.9}})
          .bindTooltip(`${{escapeHtml(station.e)}}<br><b>${{Number(row[2]).toFixed(1)}} mm</b>`)
          .bindPopup(popup(station,row,period), {{maxWidth:390}}).addTo({observations_name});
      }}
      label.textContent = period;
      const estimation = interpolationData[period];
      if (interpolationOverlay) {{ {interpolation_name}.removeLayer(interpolationOverlay); interpolationOverlay=null; }}
      if (estimation && estimation.image) {{
        interpolationOverlay=L.imageOverlay(estimation.image,{bounds_json},{{opacity:.72,interactive:false,pane:'tilePane'}});
        interpolationOverlay.addTo({interpolation_name});
      }}
      const s = periodStatistics[period];
      const rows = [['Observaciones',s.cantidad_observaciones],['Estaciones',s.cantidad_estaciones],
        ['Datasets',s.cantidad_datasets],['Fuentes',s.cantidad_fuentes],
        ['IDW',estimation && estimation.has_estimation ? `${{estimation.station_count}} estaciones · estimación` : 'Sin datos suficientes'],
        ['Mínima',`${{s.precipitacion_minima.toFixed(1)}} mm`],
        ['Máxima',`${{s.precipitacion_maxima.toFixed(1)}} mm`],
        ['Media',`${{s.precipitacion_media.toFixed(1)}} mm`],
        ['Mediana',`${{s.precipitacion_mediana.toFixed(1)}} mm`]];
      statsGrid.innerHTML = rows.map(item => `<span>${{item[0]}}</span><strong>${{item[1]}}</strong>`).join('');
    }}
    slider.addEventListener('input', event => renderPeriod(Number(event.target.value)));
    function movePeriod(delta) {{
      slider.value=Math.max(0,Math.min(periods.length-1,Number(slider.value)+delta));
      renderPeriod(Number(slider.value));
    }}
    previousButton.addEventListener('click', () => movePeriod(-1));
    nextButton.addEventListener('click', () => movePeriod(1));
    document.addEventListener('keydown', event => {{
      if (event.key==='ArrowLeft') movePeriod(-1);
      if (event.key==='ArrowRight') movePeriod(1);
    }});
    playButton.addEventListener('click', () => {{
      if (timer) {{ clearInterval(timer); timer=null; playButton.textContent='▶'; return; }}
      playButton.textContent='❚❚'; timer=setInterval(() => {{
        slider.value=(Number(slider.value)+1)%periods.length; renderPeriod(Number(slider.value));
      }},900);
    }});
    renderPeriod(Number(slider.value));
    """


def generate_map(
    frame: pd.DataFrame,
    provinces_path: Path = config.PROVINCES_GEOJSON,
    output_path: Path = config.OUTPUT_HTML,
) -> Path:
    """Genera un HTML estático con datos meteorológicos y provinciales embebidos."""
    if frame.empty:
        raise ValueError("No hay observaciones trimestrales para visualizar")
    if not provinces_path.is_file():
        raise FileNotFoundError(f"No se encontró {provinces_path}")
    maximum = precipitation_global_maximum(frame)
    generate_precipitation_ticks(maximum, config.PRECIPITATION_STEP)
    temporal_payload = build_compact_temporal_payload(frame)
    stats_payload = _statistics_payload(frame)
    territory = load_territory(provinces_path, config.TARGET_CRS)
    spatial_grid = create_spatial_grid(territory, config.GRID_RESOLUTION)
    interpolation_payload = build_interpolation_payload(frame, spatial_grid, maximum)

    map_object = folium.Map(
        location=config.ARGENTINA_CENTER,
        zoom_start=config.DEFAULT_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    observations = folium.FeatureGroup(name="Observaciones reales", show=True).add_to(map_object)
    interpolation = folium.FeatureGroup(
        name="Precipitación interpolada (estimación IDW)", show=True
    ).add_to(map_object)
    coverage = folium.FeatureGroup(name="Cobertura espacial", show=False).add_to(map_object)
    unique_stations = frame.loc[
        frame["provincia"].str.casefold().ne("sin asignar")
    ].drop_duplicates(subset=["dataset_id"])
    HeatMap(
        unique_stations[["latitud", "longitud"]].values.tolist(),
        radius=20,
        blur=16,
        min_opacity=0.25,
        name="Densidad de estaciones",
    ).add_to(coverage)
    provinces = folium.FeatureGroup(name="Límites provinciales", show=True).add_to(map_object)
    province_data = gpd.read_file(provinces_path).to_crs(config.TARGET_CRS)
    province_data["geometry"] = province_data.geometry.simplify(
        config.GEOJSON_SIMPLIFICATION_TOLERANCE, preserve_topology=True
    )
    folium.GeoJson(
        json.loads(province_data.to_json()),
        style_function=lambda _feature: {
            "color": "#37474f", "weight": 1.2, "fillOpacity": 0.02,
        },
        tooltip=folium.GeoJsonTooltip(fields=["nombre"], aliases=["Provincia:"]),
    ).add_to(provinces)
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.get_root().html.add_child(Element(_controls_html(maximum)))
    map_object.add_child(
        _DeferredScript(
            _map_script(
                observations.get_name(), interpolation.get_name(),
                temporal_payload, stats_payload, interpolation_payload, spatial_grid.bounds,
                maximum,
            )
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(output_path)
    return output_path
