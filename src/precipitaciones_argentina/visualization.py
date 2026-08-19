# ruff: noqa: E501
"""Generación del mapa temporal estático con Folium y Leaflet.

Las líneas extensas corresponden a JavaScript y CSS embebidos en el HTML estático.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import Element, MacroElement, Template
from folium.plugins import HeatMap
from PIL import Image
from shapely import intersects_xy

from . import config
from .coverage import nearest_station_distance_km, stations_within_radius
from .spatial import (
    InterpolationResult,
    SpatialGrid,
    active_stations,
    create_spatial_grid,
    idw_at_points,
    interpolate,
    load_territory,
    raster_pixel_for_coordinate,
    validate_active_station_coverage,
)
from .statistics import (
    generate_log_legend_ticks,
    generate_precipitation_ticks,
    normalize_precipitation_for_color,
    precipitation_global_maximum,
    thin_legend_ticks,
)
from .temporal import (
    ACCUMULATED_COLUMN,
    MEAN_COLUMN,
    add_climate_anomalies,
    ensure_quarterly_canonical_columns,
)

COLOR_STOPS = [
    (0.0, "#ffffd9"),
    (0.2, "#c7e9b4"),
    (0.4, "#7fcdbb"),
    (0.6, "#41b6c4"),
    (0.8, "#225ea8"),
    (1.0, "#081d58"),
]
LOGGER = logging.getLogger(__name__)


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
    frame = ensure_quarterly_canonical_columns(frame)
    if "tipo_precipitacion" not in frame:
        frame = frame.assign(tipo_precipitacion="incremental")
    station_fields = [
        "dataset_id", "archivo_origen", "fuente", "estacion", "localidad", "provincia",
        "latitud", "longitud", "unidad_original", "tipo_precipitacion",
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
            "t": row.tipo_precipitacion,
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
                round(float(getattr(row, ACCUMULATED_COLUMN)), 6),
                int(row.cantidad_observaciones),
                optional_number(getattr(row, "precipitacion_historica_mm", None)),
                optional_number(getattr(row, "anomalia_absoluta_mm", None)),
                optional_number(getattr(row, "anomalia_relativa_pct", None)),
                optional_number(getattr(row, "anios_historicos", None)),
                optional_number(getattr(row, MEAN_COLUMN, None)),
                optional_number(getattr(row, "precipitacion_minima_mm", None)),
                optional_number(getattr(row, "precipitacion_maxima_mm", None)),
                optional_number(getattr(row, ACCUMULATED_COLUMN, None)),
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
    values: np.ndarray,
    valid_mask: np.ndarray,
    maximum: float,
    alpha: int = 255,
    scale: str = "linear",
) -> np.ndarray:
    """Aplica la misma escala continua global del mapa y transparencia fuera de cobertura."""
    if scale not in {"linear", "log"}:
        raise ValueError("scale debe ser 'linear' o 'log'")
    bounded = np.clip(values, 0, maximum) if maximum > 0 else np.zeros_like(values)
    if maximum <= 0:
        ratio = np.zeros_like(values)
    elif scale == "log":
        ratio = np.log1p(bounded) / np.log1p(maximum)
    else:
        ratio = bounded / maximum
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
        # Math.round de JavaScript para canales no negativos: floor(x + 0.5).
        result[selection, :3] = np.floor(colors[selection] + 0.5).astype(np.uint8)
    result[valid_mask, 3] = alpha
    return result


def precipitation_to_rgba(
    value_mm: float,
    *,
    maximum_mm: float,
    scale: str,
    alpha: float = 1.0,
) -> tuple[int, int, int, int]:
    """Convierte un valor real con la misma ruta cromática usada por el ráster."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha debe estar entre 0 y 1")
    rgba = precipitation_rgba(
        np.array([[value_mm]], dtype=float),
        np.array([[True]]),
        maximum_mm,
        alpha=round(alpha * 255),
        scale=scale,
    )
    return tuple(int(channel) for channel in rgba[0, 0])


def anomaly_rgba(
    values: np.ndarray, valid_mask: np.ndarray, limit: float, alpha: int = 255
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


def orient_rgba_for_leaflet(rgba: np.ndarray) -> np.ndarray:
    """Orienta norte arriba; conserva oeste a la izquierda."""
    return np.flipud(rgba)


def _rgba_data_url(rgba: np.ndarray) -> str:
    """Codifica un ráster RGBA como PNG embebible."""
    buffer = io.BytesIO()
    Image.fromarray(orient_rgba_for_leaflet(rgba), mode="RGBA").save(
        buffer, format="PNG", optimize=True
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _write_spatial_debug(
    period: str, grid: SpatialGrid, result: InterpolationResult, rgba: np.ndarray
) -> None:
    """Escribe matrices diagnósticas sólo cuando SPATIAL_DEBUG está habilitado."""
    debug_directory = config.OUTPUT_DIR / "debug"
    debug_directory.mkdir(parents=True, exist_ok=True)
    final_rgba = orient_rgba_for_leaflet(rgba)
    mask = (final_rgba[..., 3] > 0).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(debug_directory / f"mask_{period}.png")
    Image.fromarray(final_rgba[..., 3], mode="L").save(
        debug_directory / f"alpha_{period}.png"
    )
    Image.fromarray(final_rgba, mode="RGBA").save(
        debug_directory / f"rgba_{period}.png"
    )
    values = np.flipud(result.values)
    finite = np.isfinite(values)
    grayscale = np.zeros(values.shape, dtype=np.uint8)
    if finite.any():
        minimum, maximum = values[finite].min(), values[finite].max()
        if maximum > minimum:
            normalized = (values[finite] - minimum) / (maximum - minimum)
            grayscale[finite] = (normalized * 255).astype(np.uint8)
    Image.fromarray(grayscale, mode="L").save(debug_directory / f"idw_{period}.png")


def _write_alignment_debug(
    period: str, grid: SpatialGrid, stations: pd.DataFrame
) -> None:
    """Genera un ráster sin IDW y su tabla de alineación estación-píxel."""
    debug_directory = config.OUTPUT_DIR / "debug"
    debug_directory.mkdir(parents=True, exist_ok=True)
    alignment = np.zeros(
        (len(grid.latitudes), len(grid.longitudes), 4), dtype=np.uint8
    )
    patterns = ["Pergamino", "Chascom"]
    colors = [(255, 0, 0, 255), (0, 170, 0, 255), (0, 0, 255, 255)]
    selected: list[pd.Series] = []
    for pattern in patterns:
        matches = stations.loc[
            stations["estacion"].str.contains(pattern, case=False, regex=True, na=False)
        ]
        if not matches.empty:
            selected.append(matches.iloc[0])
    # El período de control no posee una estación denominada Bahía Blanca.
    # Se usa la estación activa más cercana a esa ciudad como equivalente austral.
    bahia_latitude, bahia_longitude = -38.72, -62.27
    if not stations.empty:
        latitude_delta = stations["latitud"].astype(float) - bahia_latitude
        longitude_delta = (
            (stations["longitud"].astype(float) - bahia_longitude)
            * np.cos(np.deg2rad(bahia_latitude))
        )
        selected.append(stations.loc[(latitude_delta**2 + longitude_delta**2).idxmin()])
    records: list[dict[str, object]] = []
    for station, color in zip(selected, colors, strict=False):
        pixel = raster_pixel_for_coordinate(
            grid, float(station["longitud"]), float(station["latitud"])
        )
        alignment[pixel.row, pixel.column] = color
        records.append({
            "station": station["estacion"],
            "real_lat": station["latitud"], "real_lon": station["longitud"],
            "row": pixel.row, "column": pixel.column,
            "pixel_lat": pixel.center_latitude,
            "pixel_lon": pixel.center_longitude,
            "delta_lat": pixel.center_latitude - float(station["latitud"]),
            "delta_lon": pixel.center_longitude - float(station["longitud"]),
        })
    Image.fromarray(alignment, mode="RGBA").save(
        debug_directory / "raster_alignment.png"
    )
    pd.DataFrame(records).to_csv(
        debug_directory / "raster_alignment_points.csv", index=False
    )
    LOGGER.info(
        "=== RASTER GEOREFERENCE === resolución=%s shape=%s "
        "latitudes=south_to_north Leaflet=north_to_south flipud=YES "
        "bounds=%s row0_lat=%.6f last_row_lat=%.6f",
        config.GRID_RESOLUTION, alignment.shape[:2], grid.bounds,
        grid.latitudes[-1], grid.latitudes[0],
    )


def build_interpolation_payload(
    frame: pd.DataFrame,
    grid: SpatialGrid,
    maximum: float,
    anomaly_maximum: float,
    relative_limit: float,
) -> dict[str, dict[str, dict[str, object]]]:
    """Precalcula superficies IDW para precipitación y anomalías por período."""
    frame = ensure_quarterly_canonical_columns(frame)
    payload: dict[str, dict[str, dict[str, object]]] = {}
    for period, rows in frame.groupby("periodo", sort=False):
        period_payload: dict[str, dict[str, object]] = {}
        modes = {
            "absolute": (ACCUMULATED_COLUMN, maximum, precipitation_rgba),
            "anomaly_abs": ("anomalia_absoluta_mm", anomaly_maximum, anomaly_rgba),
            "anomaly_rel": (
                "anomalia_relativa_pct",
                relative_limit,
                anomaly_rgba,
            ),
        }
        precipitation_active = active_stations(rows, ACCUMULATED_COLUMN, grid.territory)
        active_ids = set(precipitation_active["dataset_id"])
        diagnostic = rows.loc[
            rows["estacion"].str.contains("Chascom", case=False, na=False)
        ]
        for station in diagnostic.itertuples(index=False):
            included = station.dataset_id in active_ids
            LOGGER.debug(
                "Estación=%s período=%s precipitación=%s lat=%s lon=%s "
                "ACTIVE_STATIONS=%s IDW=%s",
                station.estacion, period, getattr(station, ACCUMULATED_COLUMN),
                station.latitud, station.longitud,
                "YES" if included else "NO", "YES" if included else "NO",
            )
        las_armas = rows.loc[
            rows["estacion"].str.contains("Las Armas", case=False, na=False)
        ]
        for station in las_armas.itertuples(index=False):
            accumulated = float(getattr(station, ACCUMULATED_COLUMN))
            LOGGER.debug(
                "Consistencia estación=%s período=%s promedio=%.1f mm "
                "acumulado=%.1f mm gráfico=%.1f mm color=%.1f mm IDW=%.1f mm "
                "máximo_global=%.1f mm escala=log normalizado=%.8f",
                station.estacion, period, float(getattr(station, MEAN_COLUMN)),
                accumulated, accumulated, accumulated, accumulated, maximum,
                normalize_precipitation_for_color(accumulated, maximum, "log"),
            )
        for mode, (column, limit, colorizer) in modes.items():
            if column not in rows:
                period_payload[mode] = {
                    "image": None, "station_count": 0, "has_estimation": False,
                }
                continue
            active = active_stations(rows, column, grid.territory)
            validate_active_station_coverage(
                active, grid.territory, config.MAX_INTERPOLATION_DISTANCE_KM
            )
            result = interpolate(
                config.INTERPOLATION_METHOD,
                active["longitud"].to_numpy(dtype=float),
                active["latitud"].to_numpy(dtype=float),
                active[column].to_numpy(dtype=float),
                grid,
                power=config.IDW_POWER,
                maximum_distance_km=config.MAX_INTERPOLATION_DISTANCE_KM,
                minimum_stations=config.MIN_INTERPOLATION_STATIONS,
            )
            if (
                config.SPATIAL_DEBUG and mode == "absolute"
                and str(period) == config.SPATIAL_DEBUG_PERIOD
            ):
                LOGGER.info("Resumen espacial %s: %s", period, result.diagnostics)
            if (
                config.SPATIAL_DEBUG and mode == "absolute"
                and str(period) == config.SPATIAL_DEBUG_PERIOD
            ):
                chascomus = active.loc[
                    active["estacion"].str.contains("Chascom", case=False, na=False)
                ]
                for station in chascomus.itertuples(index=False):
                    exact = idw_at_points(
                        active["longitud"].to_numpy(float),
                        active["latitud"].to_numpy(float),
                        active[column].to_numpy(float),
                        np.array([station.longitud]), np.array([station.latitud]),
                        grid.territory, power=config.IDW_POWER,
                        maximum_distance_km=config.MAX_INTERPOLATION_DISTANCE_KM,
                    )
                    inside_bbox = (
                        grid.longitudes.min() <= station.longitud <= grid.longitudes.max()
                        and grid.latitudes.min() <= station.latitud <= grid.latitudes.max()
                    )
                    nearest_x = int(np.abs(grid.longitudes - station.longitud).argmin())
                    nearest_y = int(np.abs(grid.latitudes - station.latitud).argmin())
                    LOGGER.info(
                        "=== DIAGNÓSTICO ESPACIAL === estación=%s período=%s lat=%s "
                        "lon=%s acumulado=%.1f ACTIVE_STATIONS=YES IDW=YES bbox=%s "
                        "territorio=%s distancia_km=%.6f umbral=%s punto_grilla=%s "
                        "valor_idw_exacto=%.1f máscara_final=%s grid_bounds=%s",
                        station.estacion, period, station.latitud, station.longitud,
                        float(getattr(station, ACCUMULATED_COLUMN)), inside_bbox,
                        bool(exact.territory_mask[0]), exact.distances_km[0],
                        bool(exact.distance_mask[0]),
                        bool(result.valid_mask[nearest_y, nearest_x]), exact.values[0],
                        bool(exact.valid_mask[0]), grid.bounds,
                    )
            url = None
            rgba = None
            scale_images = None
            if result.valid_mask.any():
                if mode == "absolute":
                    rgba = precipitation_rgba(
                        result.values, result.valid_mask, limit, scale="linear"
                    )
                    logarithmic_rgba = precipitation_rgba(
                        result.values, result.valid_mask, limit, scale="log"
                    )
                    scale_images = {
                        "linear": _rgba_data_url(rgba),
                        "log": _rgba_data_url(logarithmic_rgba),
                    }
                    url = scale_images["linear"]
                else:
                    rgba = colorizer(result.values, result.valid_mask, limit)
                    url = _rgba_data_url(rgba)
            if (
                config.SPATIAL_DEBUG and mode == "absolute"
                and str(period) == config.SPATIAL_DEBUG_PERIOD and rgba is not None
            ):
                _write_spatial_debug(str(period), grid, result, rgba)
                _write_alignment_debug(str(period), grid, active)
            if (
                config.SPATIAL_DEBUG and mode == "absolute"
                and str(period) == config.SPATIAL_DEBUG_PERIOD and rgba is not None
            ):
                diagnostics = active.loc[
                    active["estacion"].str.contains(
                        "Chascom|Las Armas", case=False, regex=True, na=False
                    )
                ]
                for station in diagnostics.itertuples(index=False):
                    pixel = raster_pixel_for_coordinate(
                        grid, station.longitud, station.latitud
                    )
                    latitude_index = len(grid.latitudes) - 1 - pixel.row
                    final_rgba = orient_rgba_for_leaflet(rgba)
                    pixel_value = float(result.values[latitude_index, pixel.column])
                    exact = idw_at_points(
                        active["longitud"].to_numpy(float),
                        active["latitud"].to_numpy(float),
                        active[column].to_numpy(float),
                        np.array([station.longitud]), np.array([station.latitud]),
                        grid.territory, power=config.IDW_POWER,
                        maximum_distance_km=config.MAX_INTERPOLATION_DISTANCE_KM,
                    )
                    observed = float(getattr(station, ACCUMULATED_COLUMN))
                    marker_rgba = precipitation_to_rgba(
                        observed, maximum_mm=maximum, scale="log"
                    )
                    raster_rgba = precipitation_to_rgba(
                        pixel_value, maximum_mm=maximum, scale="log"
                    )
                    LOGGER.info(
                        "=== COLOR DIAGNOSTIC === estación=%s período=%s observado=%.3f "
                        "IDW_exacto=%.3f IDW_pixel=%.3f máximo_global=%.3f escala=log "
                        "normalización_lineal=%.8f normalización_log=%.8f "
                        "normalización_marcador=%.8f normalización_raster=%.8f "
                        "RGBA_marcador=%s RGBA_raster=%s alpha=%d; "
                        "row=%d column=%d centro=(%.6f, %.6f) "
                        "bounds=(W %.6f E %.6f S %.6f N %.6f) inside_country=%s "
                        "RGBA_lineal_embebido=%s",
                        station.estacion, period, observed, float(exact.values[0]),
                        pixel_value, maximum,
                        normalize_precipitation_for_color(observed, maximum, "linear"),
                        normalize_precipitation_for_color(observed, maximum, "log"),
                        normalize_precipitation_for_color(observed, maximum, "log"),
                        normalize_precipitation_for_color(pixel_value, maximum, "log"),
                        marker_rgba, raster_rgba, raster_rgba[3],
                        pixel.row, pixel.column, pixel.center_latitude,
                        pixel.center_longitude, pixel.west, pixel.east,
                        pixel.south, pixel.north,
                        bool(grid.territory_mask[latitude_index, pixel.column]),
                        tuple(int(value) for value in final_rgba[pixel.row, pixel.column]),
                    )
            period_payload[mode] = {
                "station_count": int(len(active)),
                "location_count": result.station_count,
                "has_estimation": url is not None,
            }
            if mode == "absolute":
                period_payload[mode]["images"] = scale_images
            else:
                period_payload[mode]["image"] = url
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
      #legend-panel {{ left:12px; bottom:24px; padding:10px; }}
      #legend-panel[open] {{ width:370px; }}
      #legend-panel:not([open]) {{ width:auto; }}
      #legend-panel summary {{ cursor:pointer; user-select:none; white-space:nowrap; }}
      #update-panel {{ left: 52px; top: 12px; padding:8px 11px; }}
      #analysis-panel {{ left:52px; top:108px; width:310px; padding:9px; }}
      #analysis-panel select {{ width:100%; margin:2px 0 5px; }}
      #series-chart {{ width:100%; height:145px; background:#fafafa; }}
      #series-summary {{ font-size:11px; }}
      #legend-gradient {{ height:12px; background:linear-gradient(to right,{stops}); margin:6px 0 2px; }}
      #legend-values {{ position:relative; height:48px; font-size:9px; overflow:visible; }}
      #legend-values span {{ position:absolute; top:2px; transform:translateX(-50%) rotate(-45deg);
        transform-origin:top center; white-space:nowrap; }}
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
      <small id="stats-basis">Sobre acumulados trimestrales por estación</small>
      <div id="stats-grid"></div></div>
    <div id="update-panel" class="precip-panel"><strong>Precipitaciones Argentina</strong><br>
      Última actualización: {generated_label}<br>Datasets utilizados: {datasets} · Estaciones: {stations}<br>
      Período disponible: {period_minimum} → {period_maximum}</div>
    <details id="analysis-panel" class="precip-panel"><summary><strong>Análisis avanzado</strong></summary>
      <label>Variable</label><select id="mode-select">
        <option value="absolute">Precipitación absoluta (mm)</option>
        <option value="anomaly_abs">Anomalía absoluta (mm)</option>
        <option value="anomaly_rel">Anomalía relativa (%)</option></select>
      <label>Escala de color</label><select id="color-scale-select">
        <option value="log" selected>Logarítmica</option>
        <option value="linear">Lineal</option></select>
      <label title="Muestra la superficie IDW con opacidad total para comparar directamente sus colores con la leyenda.">
        <input type="checkbox" id="idw-opaque-checkbox"> IDW sin transparencia</label><br>
      <label>Provincia</label><select id="province-filter"><option value="">Argentina completa</option></select>
      <label>Fuente</label><select id="source-filter"><option value="">Todas las fuentes</option></select>
      <label>Serie por estación</label><select id="station-select"></select>
      <label>Comparación</label><select id="comparison-quarter">
        <option value="annual" selected>Trimestres del año seleccionado</option>
        <option value="full">Serie completa</option></select>
      <div id="series-title"><strong>Precipitación acumulada trimestral</strong></div>
      <svg id="series-chart" viewBox="0 0 300 130" preserveAspectRatio="none"></svg>
      <div id="series-summary"></div>
    </details>
    <details id="legend-panel" class="precip-panel" open>
      <summary><strong id="legend-title">Precipitación acumulada trimestral (mm)</strong></summary>
      <div id="legend-gradient"></div><div id="legend-values">0 · {maximum:g} mm</div>
      <small id="legend-note">Color: escala logarítmica · valores expresados en mm</small><br>
      <small>● observado &nbsp; ≠ estimación espacial</small>
    </details>
    """


def _map_script(
    map_name: str,
    observations_name: str,
    interpolation_name: str,
    coverage_heatmap_name: str,
    payload: dict[str, object],
    interpolation_payload: dict[str, dict[str, object]],
    interpolation_bounds: list[list[float]],
    maximum: float,
    anomaly_maximum: float,
    relative_limit: float,
    linear_ticks: list[float],
    log_ticks: list[float],
) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    interpolation_json = json.dumps(
        interpolation_payload, ensure_ascii=False, separators=(",", ":")
    )
    bounds_json = json.dumps(interpolation_bounds)
    colors_json = json.dumps(COLOR_STOPS)
    linear_ticks_json = json.dumps(linear_ticks)
    log_ticks_json = json.dumps(log_ticks)
    return f"""
    const precipitationData = {data_json};
    const interpolationData = {interpolation_json};
    const precipitationMaximum = {maximum};
    const anomalyMaximum = {anomaly_maximum};
    const relativeAnomalyLimit = {relative_limit};
    const idwDefaultOpacity = {config.IDW_DEFAULT_OPACITY};
    const precipitationColors = {colors_json};
    const linearLegendTicks = {linear_ticks_json};
    const logLegendTicks = {log_ticks_json};
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
    const colorScaleSelect = document.getElementById('color-scale-select');
    const idwOpaqueCheckbox = document.getElementById('idw-opaque-checkbox');
    const provinceFilter = document.getElementById('province-filter');
    const sourceFilter = document.getElementById('source-filter');
    const stationSelect = document.getElementById('station-select');
    const comparisonQuarter = document.getElementById('comparison-quarter');
    const seriesChart = document.getElementById('series-chart');
    const seriesTitle = document.getElementById('series-title');
    const seriesSummary = document.getElementById('series-summary');
    const query = new URLSearchParams(window.location.search);
    const requestedMode = query.get('mode');
    if (['absolute','anomaly_abs','anomaly_rel'].includes(requestedMode)) modeSelect.value=requestedMode;
    const requestedScale = query.get('scale');
    if (['linear','log'].includes(requestedScale)) colorScaleSelect.value=requestedScale;
    if (query.get('opaque')==='1') idwOpaqueCheckbox.checked=true;
    const focusLat=Number(query.get('lat')), focusLon=Number(query.get('lon'));
    const focusZoom=Number(query.get('zoom'));
    if(Number.isFinite(focusLat) && Number.isFinite(focusLon))
      {map_name}.setView([focusLat,focusLon],Number.isFinite(focusZoom) ? focusZoom : 8);
    slider.max = Math.max(0, periods.length - 1);
    slider.value = Math.max(0, periods.length - 1);
    const requestedPeriod = query.get('period');
    if (periods.includes(requestedPeriod)) slider.value=periods.indexOf(requestedPeriod);
    let timer = null;
    let interpolationOverlay = null;
    let idwOpaque = idwOpaqueCheckbox.checked;

    function getCurrentIdwOpacity() {{ return idwOpaque ? 1.0 : idwDefaultOpacity; }}
    function updateIdwOpacity() {{
      idwOpaque=idwOpaqueCheckbox.checked;
      if(interpolationOverlay) interpolationOverlay.setOpacity(getCurrentIdwOpacity());
    }}

    function escapeHtml(value) {{
      return String(value ?? '—').replace(/[&<>'"]/g, char =>
        ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}})[char]);
    }}
    function normalizePrecipitationForColor(value,scale) {{
      if (precipitationMaximum <= 0) return 0;
      const bounded=Math.max(0,Math.min(precipitationMaximum,value));
      return scale==='log' ? Math.log1p(bounded)/Math.log1p(precipitationMaximum) : bounded/precipitationMaximum;
    }}
    function absoluteColor(value) {{
      const ratio = normalizePrecipitationForColor(value,colorScaleSelect.value);
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
      const statistics = !row || !row[3]
        ? '<div style="margin-top:8px;padding-top:7px;border-top:1px solid #bbb"><b>Estadísticas pluviométricas observadas</b><br>Sin registros de precipitación disponibles para este período</div>'
        : `<div style="margin-top:8px;padding-top:7px;border-top:1px solid #bbb"><b>Estadísticas pluviométricas observadas</b><table>
          <tr><td><b>Promedio de los registros</b></td><td>${{Number(row[8]).toFixed(1)}} mm</td></tr>
          <tr><td><b>Acumulado trimestral</b></td><td>${{Number(row[11]).toFixed(1)}} mm</td></tr>
          <tr><td><b>Valor utilizado para color</b></td><td>${{Number(row[2]).toFixed(1)}} mm</td></tr>
          <tr><td><b>Mínimo registrado</b></td><td>${{Number(row[9]).toFixed(1)}} mm</td></tr>
          <tr><td><b>Máximo registrado</b></td><td>${{Number(row[10]).toFixed(1)}} mm</td></tr>
          <tr><td><b>Cantidad de registros válidos</b></td><td>${{row[3]}}</td></tr></table>
          <small>Exclusivamente observaciones reales; no incluye IDW ni estaciones vecinas.</small></div>`;
      const [year,quarter] = period.split('-');
      const fields = [['Estación',station.e],['Localidad',station.l],
        ['Provincia',station.p],['Fuente',station.f],['Dataset',station.d],
        ['Archivo',station.a],['Año',year],['Trimestre',quarter],['Período',period],
        ['Latitud',station.y],['Longitud',station.x],['Tipo','Dato observado']];
      return '<b>Metadatos de la estación</b><table>' + fields.map(item => `<tr><td><b>${{escapeHtml(item[0])}}</b></td><td>${{escapeHtml(item[1])}}</td></tr>`).join('') + '</table>' + statistics;
    }}
    function renderPeriod(index) {{
      const period = periods[index]; if (!period) return;
      {observations_name}.clearLayers();
      const visibleRows=filteredRows(period); const values=[];
      const rowsByStation=new Map(visibleRows.map(row=>[row[0],row]));
      stationMetadata.forEach((station,stationIndex) => {{ if(!stationAllowed(station)) return;
        const row=rowsByStation.get(stationIndex); const value=row ? valueFor(row) : null;
        if(value != null) values.push(Number(value)); const color=value == null ? '#9e9e9e' : colorFor(Number(value));
        L.circleMarker([station.y,station.x], {{radius:6,color:'#263238',weight:.6,
          fillColor:color,fillOpacity:.9}})
          .bindTooltip(value == null ? `${{escapeHtml(station.e)}}<br><b>Sin registros para el período</b>` : `${{escapeHtml(station.e)}}<br><b>${{Number(value).toFixed(1)}} ${{unitForMode()}}</b>`)
          .bindPopup(popup(station,row,period), {{maxWidth:390}}).addTo({observations_name});
      }});
      label.textContent = period;
      const estimationSet = interpolationData[period];
      const estimation = estimationSet ? estimationSet[modeSelect.value] : null;
      if (interpolationOverlay) {{ {interpolation_name}.removeLayer(interpolationOverlay); interpolationOverlay=null; }}
      const estimationImage=estimation && modeSelect.value==='absolute' && estimation.images
        ? estimation.images[colorScaleSelect.value] : estimation ? estimation.image : null;
      if (estimationImage) {{
        interpolationOverlay=L.imageOverlay(estimationImage,{bounds_json},{{opacity:getCurrentIdwOpacity(),interactive:false,pane:'tilePane'}});
        interpolationOverlay.addTo({interpolation_name});
      }}
      values.sort((a,b)=>a-b); const mean=values.reduce((a,b)=>a+b,0)/(values.length||1);
      const median=values.length ? (values[Math.floor((values.length-1)/2)]+values[Math.ceil((values.length-1)/2)])/2 : NaN;
      const rows = [['Observaciones',visibleRows.reduce((sum,row)=>sum+row[3],0)],
        ['Estaciones',new Set(visibleRows.map(row=>row[0])).size],
        ['Datasets',new Set(visibleRows.map(row=>stationMetadata[row[0]].d)).size],
        ['Fuentes',new Set(visibleRows.map(row=>stationMetadata[row[0]].f)).size],
        ['IDW',estimation && estimation.has_estimation ? `${{estimation.station_count}} estaciones activas · ${{estimation.location_count}} ubicaciones · estimación de red completa` : 'Sin datos suficientes'],
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
      const statsBasis=document.getElementById('stats-basis');
      colorScaleSelect.disabled=modeSelect.value!=='absolute';
      if(modeSelect.value==='absolute') {{ title.textContent='Precipitación acumulada trimestral (mm)';
        statsBasis.textContent='Sobre acumulados trimestrales por estación';
        description.textContent='Precipitación acumulada trimestral · flechas ←/→ para navegar';
        const ticks=colorScaleSelect.value==='log' ? logLegendTicks : linearLegendTicks;
        labels.innerHTML=ticks.map(value=>{{ const position=normalizePrecipitationForColor(value,colorScaleSelect.value)*100;
          return `<span style="left:${{position}}%">${{Number(value).toLocaleString('es-AR',{{maximumFractionDigits:1}})}}</span>`;
        }}).join('');
        gradient.style.background='linear-gradient(to right,#ffffd9,#c7e9b4,#7fcdbb,#41b6c4,#225ea8,#081d58)';
        note.textContent=`Color: escala ${{colorScaleSelect.value==='log' ? 'logarítmica' : 'lineal'}} · valores expresados en mm`; }}
      else {{ const limit=modeSelect.value==='anomaly_abs' ? anomalyMaximum : relativeAnomalyLimit;
        statsBasis.textContent=modeSelect.value==='anomaly_abs' ? 'Sobre anomalías absolutas por estación' : 'Sobre anomalías relativas por estación';
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
      stationMetadata.map((station,index)=>({{station,index}})).filter(item=>stationAllowed(item.station))
        .sort((a,b)=>a.station.e.localeCompare(b.station.e,'es',{{sensitivity:'base'}}))
        .forEach(item=>stationSelect.add(new Option(item.station.e,item.index)));
      if([...stationSelect.options].some(option=>option.value===previous)) stationSelect.value=previous;
      renderSeries(); renderPeriod(Number(slider.value));
    }}
    function selectedPeriod() {{ return periods[Number(slider.value)] || ''; }}
    function chartGrid(minimum,maximum,unit) {{
      const top=10,bottom=100,span=maximum-minimum||1; const items=[];
      for(let index=0;index<=4;index++) {{
        const ratio=index/4,y=bottom-ratio*(bottom-top),value=minimum+ratio*span;
        items.push(`<line x1="35" y1="${{y}}" x2="292" y2="${{y}}" stroke="#d7dce0" stroke-width="0.7"/>`);
        items.push(`<text x="32" y="${{y+3}}" text-anchor="end" font-size="8" fill="#59636b">${{value.toFixed(1)}}${{unit}}</text>`);
      }}
      return items.join('');
    }}
    function updateAnnualQuarterChart(stationIndex,period) {{
      if(!period) return; const [year,activeQuarter]=period.split('-');
      const station=stationMetadata[stationIndex];
      const data=['T1','T2','T3','T4'].map((quarter,index)=>{{
        const itemPeriod=`${{year}}-${{quarter}}`;
        const row=(periodRows[itemPeriod] || []).find(item=>item[0]===stationIndex);
        return {{quarter,period:itemPeriod,value:row && row[2] != null ? Number(row[2]) : null,index}};
      }});
      const available=data.filter(item=>item.value != null); const values=available.map(item=>item.value);
      const maximum=Math.max(...values,1); const x=index=>43+index*80; const y=value=>100-value/maximum*90;
      const segments=[]; for(let index=0;index<3;index++) {{ const a=data[index],b=data[index+1];
        if(a.value != null && b.value != null) segments.push(`<line x1="${{x(index)}}" y1="${{y(a.value)}}" x2="${{x(index+1)}}" y2="${{y(b.value)}}" stroke="#225ea8" stroke-width="2"/>`); }}
      const points=data.map(item=>{{ const active=item.quarter===activeQuarter;
        if(item.value == null) return `<g><line x1="${{x(item.index)-4}}" y1="88" x2="${{x(item.index)+4}}" y2="96" stroke="#999"/><line x1="${{x(item.index)+4}}" y1="88" x2="${{x(item.index)-4}}" y2="96" stroke="#999"/><title>${{item.period}} · Sin datos</title></g>`;
        return `<circle cx="${{x(item.index)}}" cy="${{y(item.value)}}" r="${{active?6:4}}" fill="#225ea8" stroke="${{active?'#f57c00':'#fff'}}" stroke-width="${{active?3:1}}"><title>Estación: ${{escapeHtml(station.e)}} · Período: ${{item.period}} · Precipitación acumulada: ${{item.value.toFixed(1)}} mm</title></circle>`;
      }}).join('');
      const labels=data.map(item=>`<text x="${{x(item.index)}}" y="118" text-anchor="middle" font-size="10" fill="${{item.quarter===activeQuarter?'#f57c00':'#333'}}" font-weight="${{item.quarter===activeQuarter?'bold':'normal'}}">${{item.quarter}}</text>`).join('');
      seriesChart.innerHTML=`${{chartGrid(0,maximum,' mm')}}<line x1="35" y1="100" x2="292" y2="100" stroke="#777"/>${{segments.join('')}}${{points}}${{labels}}`;
      seriesTitle.innerHTML=`<strong>${{escapeHtml(station.e)}}</strong><br>Precipitación acumulada trimestral — Año: ${{year}}`;
      if(!values.length) seriesSummary.textContent=`${{year}} · 0 de 4 trimestres con datos · Sin datos`;
      else {{ const min=Math.min(...values),max=Math.max(...values), count=values.length;
        seriesSummary.textContent=count===4 ? `${{year}} · T1–T4 · mín ${{min.toFixed(1)}} · máx ${{max.toFixed(1)}} mm` : `${{year}} · ${{count}} de 4 trimestres con datos · mín ${{min.toFixed(1)}} · máx ${{max.toFixed(1)}} mm`; }}
    }}
    function renderSeries() {{
      const stationIndex=Number(stationSelect.value); if(!Number.isFinite(stationIndex)) return;
      if(comparisonQuarter.value==='annual') {{ updateAnnualQuarterChart(stationIndex,selectedPeriod()); return; }}
      const data=[]; periods.forEach((period,index)=>{{ const row=periodRows[period].find(item=>item[0]===stationIndex);
        if(row) {{ const value=valueFor(row); if(value!=null) data.push([index,Number(value),period]); }} }});
      if(!data.length) {{ seriesChart.innerHTML=''; seriesSummary.textContent='Sin datos para la selección'; return; }}
      const vals=data.map(d=>d[1]), rawMin=Math.min(...vals), rawMax=Math.max(...vals);
      const min=modeSelect.value==='absolute' ? 0 : Math.min(0,rawMin), max=Math.max(rawMax,min+1), span=max-min;
      const points=data.map((d,i)=>`${{i/(data.length-1||1)*250+38}},${{100-(d[1]-min)/span*90}}`).join(' ');
      seriesChart.innerHTML=`${{chartGrid(min,max,' '+unitForMode())}}<polyline points="${{points}}" fill="none" stroke="#225ea8" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
      seriesTitle.innerHTML=`<strong>${{escapeHtml(stationMetadata[stationIndex].e)}}</strong><br>Serie histórica · ${{modeSelect.options[modeSelect.selectedIndex].text}}`;
      seriesSummary.textContent=`${{data[0][2]}} → ${{data.at(-1)[2]}} · mín ${{rawMin.toFixed(1)}} · máx ${{rawMax.toFixed(1)}} ${{unitForMode()}}`;
    }}
    slider.addEventListener('input', event => {{renderPeriod(Number(event.target.value));renderSeries();}});
    modeSelect.addEventListener('change',()=>{{updateLegend();renderSeries();renderPeriod(Number(slider.value));}});
    colorScaleSelect.addEventListener('change',()=>{{updateLegend();renderPeriod(Number(slider.value));}});
    idwOpaqueCheckbox.addEventListener('change',updateIdwOpacity);
    provinceFilter.addEventListener('change',updateStations); sourceFilter.addEventListener('change',updateStations);
    stationSelect.addEventListener('change',renderSeries); comparisonQuarter.addEventListener('change',renderSeries);
    function movePeriod(delta) {{
      slider.value=Math.max(0,Math.min(periods.length-1,Number(slider.value)+delta));
      renderPeriod(Number(slider.value)); renderSeries();
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
        slider.value=(Number(slider.value)+1)%periods.length; renderPeriod(Number(slider.value)); renderSeries();
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
    frame = ensure_quarterly_canonical_columns(frame)
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
    linear_ticks = thin_legend_ticks(
        generate_precipitation_ticks(maximum, config.PRECIPITATION_STEP)
    )
    log_ticks = generate_log_legend_ticks(maximum)
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
            "crs_distancias": config.DISTANCE_CRS,
            "minimo_estaciones": config.MIN_INTERPOLATION_STATIONS,
            "periodos_con_estimacion": estimated_periods,
            "periodos_sin_datos_suficientes": len(interpolation_payload)
            - estimated_periods,
            "mascara_territorial": True,
            "mascara_convex_hull": False,
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
    all_stations = frame.drop_duplicates(subset=["dataset_id"]).copy()
    finite_coordinates = np.isfinite(all_stations["longitud"]) & np.isfinite(
        all_stations["latitud"]
    )
    inside_territory = intersects_xy(
        territory,
        all_stations["longitud"].to_numpy(float),
        all_stations["latitud"].to_numpy(float),
    )
    unique_stations = all_stations.loc[finite_coordinates & inside_territory]
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
                map_object.get_name(),
                observations.get_name(), interpolation.get_name(), coverage_heatmap.get_name(),
                temporal_payload, interpolation_payload, spatial_grid.bounds,
                maximum, anomaly_maximum, config.RELATIVE_ANOMALY_COLOR_LIMIT,
                linear_ticks, log_ticks,
            )
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(output_path)
    return output_path
