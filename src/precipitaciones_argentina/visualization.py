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
from .coverage import nearest_station_distance_km, stations_within_radius
from .spatial import SpatialGrid, create_spatial_grid, interpolate, load_territory
from .statistics import (
    generate_precipitation_ticks,
    precipitation_global_maximum,
)
from .temporal import add_climate_anomalies

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
    periods: dict[str, list[list[object]]] = {}

    def optional_number(value: object) -> float | int | None:
        return None if pd.isna(value) else round(float(value), 6)

    for period, rows in frame.groupby("periodo", sort=False):
        periods[str(period)] = [
            [
                station_index[row.dataset_id],
                round(float(row.precipitacion_original), 6),
                round(float(row.precipitacion_mm), 6),
                int(row.cantidad_observaciones),
                optional_number(getattr(row, "precipitacion_historica_mm", None)),
                optional_number(getattr(row, "anomalia_absoluta_mm", None)),
                optional_number(getattr(row, "anomalia_relativa_pct", None)),
                optional_number(getattr(row, "anios_historicos", None)),
            ]
            for row in rows.itertuples(index=False)
        ]
    return {
        "stations": station_records,
        "periods": dict(sorted(periods.items(), key=lambda item: _period_key(item[0]))),
    }


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


def anomaly_rgba(
    values: np.ndarray, valid_mask: np.ndarray, limit: float, alpha: int = 175
) -> np.ndarray:
    """Aplica escala divergente azul-blanco-rojo centrada exactamente en cero."""
    if limit <= 0:
        raise ValueError("El límite de anomalía debe ser positivo")
    ratio = np.nan_to_num(np.clip(values / limit, -1, 1), nan=0.0)
    result = np.zeros((*values.shape, 4), dtype=np.uint8)
    negative = valid_mask & (ratio < 0)
    positive = valid_mask & ~negative
    negative_channel = (255 * (1 + ratio)).astype(np.uint8)
    positive_channel = (255 * (1 - ratio)).astype(np.uint8)
    result[negative, 0] = negative_channel[negative]
    result[negative, 1] = negative_channel[negative]
    result[negative, 2] = 255
    result[positive, 0] = 255
    result[positive, 1] = positive_channel[positive]
    result[positive, 2] = positive_channel[positive]
    result[valid_mask, 3] = alpha
    return result


def coverage_rgba(
    grid: SpatialGrid, station_coordinates: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Representa distancia y densidad sin utilizar precipitación como peso."""
    grid_x, grid_y = np.meshgrid(grid.longitudes, grid.latitudes)
    targets = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances = nearest_station_distance_km(station_coordinates, targets).reshape(grid_x.shape)
    counts = stations_within_radius(
        station_coordinates, targets, config.COVERAGE_RADIUS_KM
    ).reshape(grid_x.shape)
    ratio = np.clip(distances / config.MAX_INTERPOLATION_DISTANCE_KM, 0, 1)
    rgba = np.zeros((*grid_x.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (255 * ratio).astype(np.uint8)
    rgba[..., 1] = (180 * (1 - ratio)).astype(np.uint8)
    rgba[..., 2] = 40
    rgba[grid.territory_mask, 3] = 145
    metrics: dict[str, float | int] = {
        "distancia_media_km": float(distances[grid.territory_mask].mean()),
        "distancia_maxima_km": float(distances[grid.territory_mask].max()),
        "celdas_sin_estaciones_en_radio": int(
            (counts[grid.territory_mask] == 0).sum()
        ),
    }
    return rgba, metrics


def _rgba_data_url(rgba: np.ndarray) -> str:
    """Codifica un ráster RGBA como PNG embebible."""
    buffer = io.BytesIO()
    Image.fromarray(np.flipud(rgba), mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_interpolation_payload(
    frame: pd.DataFrame,
    grid: SpatialGrid,
    maximum: float,
    anomaly_maximum: float,
    relative_limit: float,
) -> dict[str, dict[str, dict[str, object]]]:
    """Precalcula superficies IDW para precipitación y anomalías por período."""
    payload: dict[str, dict[str, dict[str, object]]] = {}
    for period, rows in frame.groupby("periodo", sort=False):
        rows = rows.loc[rows["provincia"].str.casefold().ne("sin asignar")]
        period_payload: dict[str, dict[str, object]] = {}
        modes = {
            "absolute": ("precipitacion_mm", maximum, precipitation_rgba),
            "anomaly_abs": ("anomalia_absoluta_mm", anomaly_maximum, anomaly_rgba),
            "anomaly_rel": (
                "anomalia_relativa_pct",
                relative_limit,
                anomaly_rgba,
            ),
        }
        for mode, (column, limit, colorizer) in modes.items():
            if column not in rows:
                period_payload[mode] = {
                    "image": None, "station_count": 0, "has_estimation": False,
                }
                continue
            result = interpolate(
                config.INTERPOLATION_METHOD,
                rows["longitud"].to_numpy(dtype=float),
                rows["latitud"].to_numpy(dtype=float),
                rows[column].to_numpy(dtype=float),
                grid,
                power=config.IDW_POWER,
                maximum_distance_km=config.MAX_INTERPOLATION_DISTANCE_KM,
                minimum_stations=config.MIN_INTERPOLATION_STATIONS,
            )
            url = None
            if result.valid_mask.any():
                url = _rgba_data_url(colorizer(result.values, result.valid_mask, limit))
            period_payload[mode] = {
                "image": url,
                "station_count": result.station_count,
                "has_estimation": url is not None,
            }
        payload[str(period)] = period_payload
    return dict(sorted(payload.items(), key=lambda item: _period_key(item[0])))


def _controls_html(maximum: float, audit: dict[str, object] | None = None) -> str:
    stops = ", ".join(color for _, color in COLOR_STOPS)
    audit = audit or {}
    generated = str(audit.get("fecha_generacion", ""))[:10]
    generated_label = "—"
    if len(generated) == 10:
        year, month, day = generated.split("-")
        generated_label = f"{day}/{month}/{year}"
    datasets = audit.get("datasets_procesados", "—")
    stations = audit.get("estaciones", "—")
    period_minimum = audit.get("periodo_minimo", "—")
    period_maximum = audit.get("periodo_maximo", "—")
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
      #update-panel {{ left: 52px; top: 12px; padding:8px 11px; }}
      #analysis-panel {{ left:52px; top:108px; width:310px; padding:9px; }}
      #analysis-panel select {{ width:100%; margin:2px 0 5px; }}
      #series-chart {{ width:100%; height:120px; background:#fafafa; }}
      #series-summary {{ font-size:11px; }}
      #legend-gradient {{ height:12px; background:linear-gradient(to right,{stops}); margin:6px 0 2px; }}
      #legend-values {{ display:flex; justify-content:space-between; }}
      .leaflet-popup-content table {{ border-collapse:collapse; }}
      .leaflet-popup-content td {{ padding:2px 5px; border-bottom:1px solid #eee; }}
      @media(max-width:700px) {{ #stats-panel {{ top:180px; width:190px; }}
        #legend-panel {{ display:none; }} #time-panel {{ width:88vw; }}
        #update-panel {{ top:82px; left:10px; }} #analysis-panel {{ display:none; }} }}
    </style>
    <div id="time-panel" class="precip-panel">
      <div id="time-row"><button id="play-period" title="Reproducir">▶</button>
      <button id="previous-period" title="Período anterior">◀</button>
      <input id="period-slider" type="range" min="0" value="0" step="1">
      <button id="next-period" title="Período siguiente">▶</button>
      <span id="period-label">—</span></div>
      <div id="time-description">Precipitación acumulada trimestral · flechas ←/→ para navegar</div>
    </div>
    <div id="stats-panel" class="precip-panel"><h4>Estadísticas del período</h4>
      <div id="stats-grid"></div></div>
    <div id="update-panel" class="precip-panel"><strong>Precipitaciones Argentina</strong><br>
      Última actualización: {generated_label}<br>Datasets utilizados: {datasets} · Estaciones: {stations}<br>
      Período disponible: {period_minimum} → {period_maximum}</div>
    <details id="analysis-panel" class="precip-panel"><summary><strong>Análisis avanzado</strong></summary>
      <label>Variable</label><select id="mode-select">
        <option value="absolute">Precipitación absoluta (mm)</option>
        <option value="anomaly_abs">Anomalía absoluta (mm)</option>
        <option value="anomaly_rel">Anomalía relativa (%)</option></select>
      <label>Provincia</label><select id="province-filter"><option value="">Argentina completa</option></select>
      <label>Fuente</label><select id="source-filter"><option value="">Todas las fuentes</option></select>
      <label>Serie por estación</label><select id="station-select"></select>
      <label>Comparación</label><select id="comparison-quarter"><option value="">Serie completa</option>
        <option>T1</option><option>T2</option><option>T3</option><option>T4</option></select>
      <svg id="series-chart" viewBox="0 0 300 120" preserveAspectRatio="none"></svg>
      <div id="series-summary"></div>
    </details>
    <div id="legend-panel" class="precip-panel"><strong id="legend-title">Precipitación observada (mm)</strong>
      <div id="legend-gradient"></div><div id="legend-values"><span>0</span>
      <span id="legend-maximum">{maximum:g}</span></div><small id="legend-note">Escala global · cortes cada 10 mm</small><br>
      <small>● observado &nbsp; ≠ estimación espacial</small></div>
    """


def _map_script(
    observations_name: str,
    interpolation_name: str,
    coverage_heatmap_name: str,
    payload: dict[str, object],
    interpolation_payload: dict[str, dict[str, object]],
    interpolation_bounds: list[list[float]],
    maximum: float,
    anomaly_maximum: float,
    relative_limit: float,
) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    interpolation_json = json.dumps(
        interpolation_payload, ensure_ascii=False, separators=(",", ":")
    )
    bounds_json = json.dumps(interpolation_bounds)
    colors_json = json.dumps(COLOR_STOPS)
    return f"""
    const precipitationData = {data_json};
    const interpolationData = {interpolation_json};
    const precipitationMaximum = {maximum};
    const anomalyMaximum = {anomaly_maximum};
    const relativeAnomalyLimit = {relative_limit};
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
    const modeSelect = document.getElementById('mode-select');
    const provinceFilter = document.getElementById('province-filter');
    const sourceFilter = document.getElementById('source-filter');
    const stationSelect = document.getElementById('station-select');
    const comparisonQuarter = document.getElementById('comparison-quarter');
    const seriesChart = document.getElementById('series-chart');
    const seriesSummary = document.getElementById('series-summary');
    const query = new URLSearchParams(window.location.search);
    const requestedMode = query.get('mode');
    if (['absolute','anomaly_abs','anomaly_rel'].includes(requestedMode)) modeSelect.value=requestedMode;
    slider.max = Math.max(0, periods.length - 1);
    slider.value = Math.max(0, periods.length - 1);
    let timer = null;
    let interpolationOverlay = null;

    function escapeHtml(value) {{
      return String(value ?? '—').replace(/[&<>'"]/g, char =>
        ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}})[char]);
    }}
    function absoluteColor(value) {{
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
    function anomalyColor(value, limit) {{
      const ratio=Math.max(-1,Math.min(1,value/limit));
      if (ratio < 0) {{ const c=Math.round(255*(1+ratio)); return `rgb(${{c}},${{c}},255)`; }}
      const c=Math.round(255*(1-ratio)); return `rgb(255,${{c}},${{c}})`;
    }}
    function valueFor(row) {{
      return modeSelect.value==='absolute' ? row[2] : modeSelect.value==='anomaly_abs' ? row[5] : row[6];
    }}
    function unitForMode() {{ return modeSelect.value==='anomaly_rel' ? '%' : 'mm'; }}
    function stationAllowed(station) {{
      return (!provinceFilter.value || station.p===provinceFilter.value) &&
        (!sourceFilter.value || station.f===sourceFilter.value);
    }}
    function filteredRows(period) {{
      return periodRows[period].filter(row => stationAllowed(stationMetadata[row[0]]));
    }}
    function colorFor(value) {{
      return modeSelect.value==='absolute' ? absoluteColor(value) : anomalyColor(
        value, modeSelect.value==='anomaly_abs' ? anomalyMaximum : relativeAnomalyLimit);
    }}
    function popup(station,row,period) {{
      const [year,quarter] = period.split('-');
      const fields = [['Estación',station.e],['Localidad',station.l],
        ['Provincia',station.p],['Fuente',station.f],['Dataset',station.d],
        ['Archivo',station.a],['Año',year],['Trimestre',quarter],['Período',period],
        ['Precipitación original',`${{row[1]}} ${{station.u}}`],
        ['Precipitación en mm',`${{Number(row[2]).toFixed(1)}} mm`],
        ['Normal histórica',row[4] == null ? 'Sin normal suficiente' : `${{Number(row[4]).toFixed(1)}} mm`],
        ['Años históricos',row[7] ?? '—'],
        ['Anomalía absoluta',row[5] == null ? '—' : `${{Number(row[5]).toFixed(1)}} mm`],
        ['Anomalía relativa',row[6] == null ? '—' : `${{Number(row[6]).toFixed(1)}} %`],
        ['Latitud',station.y],['Longitud',station.x],['Tipo','Dato observado']];
      return '<table>' + fields.map(item => `<tr><td><b>${{escapeHtml(item[0])}}</b></td><td>${{escapeHtml(item[1])}}</td></tr>`).join('') + '</table>';
    }}
    function renderPeriod(index) {{
      const period = periods[index]; if (!period) return;
      {observations_name}.clearLayers();
      const visibleRows=filteredRows(period); const values=[];
      for (const row of visibleRows) {{
        const station = stationMetadata[row[0]]; const value=valueFor(row); if (value == null) continue;
        values.push(Number(value)); const color = colorFor(Number(value));
        L.circleMarker([station.y,station.x], {{radius:6,color:'#263238',weight:.6,
          fillColor:color,fillOpacity:.9}})
          .bindTooltip(`${{escapeHtml(station.e)}}<br><b>${{Number(value).toFixed(1)}} ${{unitForMode()}}</b>`)
          .bindPopup(popup(station,row,period), {{maxWidth:390}}).addTo({observations_name});
      }}
      label.textContent = period;
      const estimationSet = interpolationData[period];
      const estimation = estimationSet ? estimationSet[modeSelect.value] : null;
      if (interpolationOverlay) {{ {interpolation_name}.removeLayer(interpolationOverlay); interpolationOverlay=null; }}
      if (estimation && estimation.image) {{
        interpolationOverlay=L.imageOverlay(estimation.image,{bounds_json},{{opacity:.72,interactive:false,pane:'tilePane'}});
        interpolationOverlay.addTo({interpolation_name});
      }}
      values.sort((a,b)=>a-b); const mean=values.reduce((a,b)=>a+b,0)/(values.length||1);
      const median=values.length ? (values[Math.floor((values.length-1)/2)]+values[Math.ceil((values.length-1)/2)])/2 : NaN;
      const rows = [['Observaciones',visibleRows.reduce((sum,row)=>sum+row[3],0)],
        ['Estaciones',new Set(visibleRows.map(row=>row[0])).size],
        ['Datasets',new Set(visibleRows.map(row=>stationMetadata[row[0]].d)).size],
        ['Fuentes',new Set(visibleRows.map(row=>stationMetadata[row[0]].f)).size],
        ['IDW',estimation && estimation.has_estimation ? `${{estimation.station_count}} estaciones · estimación de red completa` : 'Sin datos suficientes'],
        ['Mínima',values.length ? `${{values[0].toFixed(1)}} ${{unitForMode()}}` : '—'],
        ['Máxima',values.length ? `${{values.at(-1).toFixed(1)}} ${{unitForMode()}}` : '—'],
        ['Media',values.length ? `${{mean.toFixed(1)}} ${{unitForMode()}}` : '—'],
        ['Mediana',values.length ? `${{median.toFixed(1)}} ${{unitForMode()}}` : '—']];
      statsGrid.innerHTML = rows.map(item => `<span>${{item[0]}}</span><strong>${{item[1]}}</strong>`).join('');
    }}
    function updateLegend() {{
      const title=document.getElementById('legend-title'), labels=document.getElementById('legend-values');
      const gradient=document.getElementById('legend-gradient'), note=document.getElementById('legend-note');
      const description=document.getElementById('time-description');
      if(modeSelect.value==='absolute') {{ title.textContent='Precipitación observada (mm)';
        description.textContent='Precipitación acumulada trimestral · flechas ←/→ para navegar';
        labels.innerHTML=`<span>0</span><span>${{precipitationMaximum}}</span>`;
        gradient.style.background='linear-gradient(to right,#ffffd9,#c7e9b4,#7fcdbb,#41b6c4,#225ea8,#081d58)'; note.textContent='Escala global · cortes cada 10 mm'; }}
      else {{ const limit=modeSelect.value==='anomaly_abs' ? anomalyMaximum : relativeAnomalyLimit;
        description.textContent=(modeSelect.value==='anomaly_abs' ? 'Anomalía absoluta trimestral' : 'Anomalía relativa trimestral')+' · flechas ←/→ para navegar';
        title.textContent=modeSelect.value==='anomaly_abs' ? 'Anomalía absoluta (mm)' : 'Anomalía relativa (%)';
        labels.innerHTML=`<span>−${{limit.toFixed(0)}}</span><span>0</span><span>+${{limit.toFixed(0)}}</span>`;
        gradient.style.background='linear-gradient(to right,#0000ff,#ffffff,#ff0000)';
        note.textContent=modeSelect.value==='anomaly_rel' ? 'Escala divergente · saturada fuera del rango' : 'Escala divergente global'; }}
    }}
    function populateFilters() {{
      [...new Set(stationMetadata.map(s=>s.p))].sort().forEach(value=>provinceFilter.add(new Option(value,value)));
      [...new Set(stationMetadata.map(s=>s.f))].sort().forEach(value=>sourceFilter.add(new Option(value,value)));
    }}
    function updateStations() {{
      const previous=stationSelect.value; stationSelect.innerHTML='';
      stationMetadata.forEach((station,index)=>{{ if(stationAllowed(station)) stationSelect.add(new Option(station.e,index)); }});
      if([...stationSelect.options].some(option=>option.value===previous)) stationSelect.value=previous;
      {coverage_heatmap_name}.setLatLngs(stationMetadata.filter(stationAllowed).map(s=>[s.y,s.x,1]));
      renderSeries(); renderPeriod(Number(slider.value));
    }}
    function renderSeries() {{
      const stationIndex=Number(stationSelect.value), quarter=comparisonQuarter.value; if(!Number.isFinite(stationIndex)) return;
      const data=[]; periods.forEach((period,index)=>{{ const row=periodRows[period].find(item=>item[0]===stationIndex);
        if(row && (!quarter || period.endsWith(quarter))) {{ const value=valueFor(row); if(value!=null) data.push([index,Number(value),period]); }} }});
      if(!data.length) {{ seriesChart.innerHTML=''; seriesSummary.textContent='Sin datos para la selección'; return; }}
      const vals=data.map(d=>d[1]), min=Math.min(...vals), max=Math.max(...vals), span=max-min||1;
      const points=data.map((d,i)=>`${{i/(data.length-1||1)*296+2}},${{116-(d[1]-min)/span*110}}`).join(' ');
      seriesChart.innerHTML=`<polyline points="${{points}}" fill="none" stroke="#225ea8" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
      seriesSummary.textContent=`${{data[0][2]}} → ${{data.at(-1)[2]}} · mín ${{min.toFixed(1)}} · máx ${{max.toFixed(1)}} ${{unitForMode()}}`;
    }}
    slider.addEventListener('input', event => renderPeriod(Number(event.target.value)));
    modeSelect.addEventListener('change',()=>{{updateLegend();renderSeries();renderPeriod(Number(slider.value));}});
    provinceFilter.addEventListener('change',updateStations); sourceFilter.addEventListener('change',updateStations);
    stationSelect.addEventListener('change',renderSeries); comparisonQuarter.addEventListener('change',renderSeries);
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
    populateFilters();
    for (const [element,value] of [[provinceFilter,query.get('province')],[sourceFilter,query.get('source')]]) {{
      if(value && [...element.options].some(option=>option.value===value)) element.value=value;
    }}
    updateLegend(); updateStations();
    const requestedStation=query.get('station');
    if(requestedStation) {{ const option=[...stationSelect.options].find(item=>stationMetadata[Number(item.value)].d===requestedStation);
      if(option) stationSelect.value=option.value; }}
    const requestedQuarter=query.get('quarter');
    if(requestedQuarter && [...comparisonQuarter.options].some(option=>option.value===requestedQuarter)) comparisonQuarter.value=requestedQuarter;
    if(query.get('advanced')==='1') document.getElementById('analysis-panel').open=true;
    renderSeries();
    """


def generate_map(
    frame: pd.DataFrame,
    provinces_path: Path = config.PROVINCES_GEOJSON,
    output_path: Path = config.OUTPUT_HTML,
    audit: dict[str, object] | None = None,
) -> Path:
    """Genera un HTML estático con datos meteorológicos y provinciales embebidos."""
    if frame.empty:
        raise ValueError("No hay observaciones trimestrales para visualizar")
    if not provinces_path.is_file():
        raise FileNotFoundError(f"No se encontró {provinces_path}")
    if "anomalia_absoluta_mm" not in frame:
        frame = add_climate_anomalies(
            frame,
            start_year=config.CLIMATOLOGY_START_YEAR,
            end_year=config.CLIMATOLOGY_END_YEAR,
            minimum_years=config.MIN_HISTORICAL_YEARS,
        )
    maximum = precipitation_global_maximum(frame)
    generate_precipitation_ticks(maximum, config.PRECIPITATION_STEP)
    anomaly_maximum = float(frame["anomalia_absoluta_mm"].abs().max())
    if pd.isna(anomaly_maximum) or anomaly_maximum == 0:
        anomaly_maximum = 1.0
    temporal_payload = build_compact_temporal_payload(frame)
    territory = load_territory(provinces_path, config.TARGET_CRS)
    spatial_grid = create_spatial_grid(territory, config.GRID_RESOLUTION)
    interpolation_payload = build_interpolation_payload(
        frame,
        spatial_grid,
        maximum,
        anomaly_maximum,
        config.RELATIVE_ANOMALY_COLOR_LIMIT,
    )
    if audit is not None:
        estimated_periods = sum(
            bool(item["absolute"]["has_estimation"])
            for item in interpolation_payload.values()
        )
        audit["interpolacion"] = {
            "metodo": config.INTERPOLATION_METHOD,
            "potencia_idw": config.IDW_POWER,
            "resolucion_grados": config.GRID_RESOLUTION,
            "distancia_maxima_km": config.MAX_INTERPOLATION_DISTANCE_KM,
            "minimo_estaciones": config.MIN_INTERPOLATION_STATIONS,
            "periodos_con_estimacion": estimated_periods,
            "periodos_sin_datos_suficientes": len(interpolation_payload)
            - estimated_periods,
            "mascara_territorial": True,
            "mascara_convex_hull": True,
            "fuera_de_cobertura_transparente": True,
        }

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
    station_coordinates = unique_stations[["longitud", "latitud"]].to_numpy(float)
    distance_rgba, coverage_metrics = coverage_rgba(spatial_grid, station_coordinates)
    if audit is not None:
        audit["cobertura_espacial"] = {
            "tipo": "HeatMap de estaciones",
            "ponderada_por_precipitacion": False,
            "estaciones_incluidas": int(len(unique_stations)),
            "radio_densidad_km": config.COVERAGE_RADIUS_KM,
            **coverage_metrics,
        }
    coverage_heatmap = HeatMap(
        unique_stations[["latitud", "longitud"]].values.tolist(),
        radius=20,
        blur=16,
        min_opacity=0.25,
        name="Densidad de estaciones",
    ).add_to(coverage)
    distance_layer = folium.FeatureGroup(
        name="Distancia a observación (red completa)", show=False
    ).add_to(map_object)
    folium.raster_layers.ImageOverlay(
        image=_rgba_data_url(distance_rgba),
        bounds=spatial_grid.bounds,
        opacity=0.65,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(distance_layer)
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
    map_object.get_root().header.add_child(
        Element(
            '<link rel="icon" href="data:image/svg+xml,'
            '<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22>'
            '<text y=%2214%22>☔</text></svg>">'
        )
    )
    map_object.get_root().html.add_child(Element(_controls_html(maximum, audit)))
    map_object.add_child(
        _DeferredScript(
            _map_script(
                observations.get_name(), interpolation.get_name(), coverage_heatmap.get_name(),
                temporal_payload, interpolation_payload, spatial_grid.bounds,
                maximum, anomaly_maximum, config.RELATIVE_ANOMALY_COLOR_LIMIT,
            )
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(output_path)
    return output_path
