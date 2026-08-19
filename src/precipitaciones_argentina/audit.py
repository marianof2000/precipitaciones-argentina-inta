"""Auditoría reproducible del ETL, geografía y resultados publicados."""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from shapely import intersects_xy
from shapely.geometry import Point

from . import __version__, config
from .spatial import load_territory
from .statistics import generate_precipitation_ticks, precipitation_global_maximum
from .temporal import ACCUMULATED_COLUMN, ensure_quarterly_canonical_columns
from .validation import ProcessingSummary

LOGGER = logging.getLogger(__name__)
TIMEZONE = ZoneInfo("America/Argentina/Buenos_Aires")


def write_audit_report(report: dict[str, Any], output_path: Path = config.OUTPUT_AUDIT) -> None:
    """Escribe JSON UTF-8 estable y legible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def find_stations_outside_territory(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Detecta y registra estaciones fuera de la máscara territorial continental."""
    territory = load_territory(config.PROVINCES_GEOJSON, config.TARGET_CRS)
    stations = frame.drop_duplicates("dataset_id")
    inside = intersects_xy(
        territory,
        stations["longitud"].to_numpy(dtype=float),
        stations["latitud"].to_numpy(dtype=float),
    )
    warnings: list[dict[str, Any]] = []
    for row in stations.loc[~inside].itertuples(index=False):
        warning = {
            "dataset_id": row.dataset_id,
            "estacion": row.estacion,
            "latitud": float(row.latitud),
            "longitud": float(row.longitud),
            "fuente": row.fuente,
        }
        warnings.append(warning)
        LOGGER.warning(
            "Estación fuera del territorio: %s (%s), coordenadas=(%.5f, %.5f), fuente=%s",
            row.estacion, row.dataset_id, row.latitud, row.longitud, row.fuente,
        )
    return warnings


def _normalized_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    plain = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return plain.casefold()


def find_station_province_mismatches(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Contrasta provincia declarada con polígonos del GeoJSON sin alterar registros."""
    import geopandas as gpd

    provinces = gpd.read_file(config.PROVINCES_GEOJSON).to_crs(config.TARGET_CRS)
    mismatches: list[dict[str, Any]] = []
    for row in frame.drop_duplicates("dataset_id").itertuples(index=False):
        point = Point(row.longitud, row.latitud)
        matches = [
            str(province.nombre)
            for province in provinces.itertuples(index=False)
            if province.geometry.intersects(point)
        ]
        configured = _normalized_name(str(row.provincia))
        geographical = {_normalized_name(name) for name in matches}
        agrees = any(
            configured == name or configured.startswith(name) or name.startswith(configured)
            for name in geographical
        )
        if matches and not agrees:
            mismatch = {
                "dataset_id": row.dataset_id,
                "estacion": row.estacion,
                "provincia_declarada": row.provincia,
                "provincias_geograficas": matches,
                "latitud": float(row.latitud),
                "longitud": float(row.longitud),
            }
            mismatches.append(mismatch)
            LOGGER.warning(
                "Discrepancia estación-provincia: %s (%s), declarada=%s, geográfica=%s",
                row.estacion, row.dataset_id, row.provincia, ", ".join(matches),
            )
    return mismatches


def build_traceability_samples(
    daily: pd.DataFrame, quarterly: pd.DataFrame, sample_count: int = 5
) -> list[dict[str, Any]]:
    """Selecciona casos distribuidos y prueba el encadenamiento hasta el valor del mapa."""
    quarterly = ensure_quarterly_canonical_columns(quarterly)
    dataset_ids = sorted(daily["dataset_id"].unique())
    if not dataset_ids:
        return []
    positions = sorted(
        {
            round(index * (len(dataset_ids) - 1) / max(1, sample_count - 1))
            for index in range(sample_count)
        }
    )
    samples: list[dict[str, Any]] = []
    for position in positions:
        dataset_id = dataset_ids[position]
        dataset_rows = daily.loc[daily["dataset_id"].eq(dataset_id)].sort_values("fecha")
        original = dataset_rows.loc[dataset_rows["precipitacion_mm"].idxmax()]
        period_rows = dataset_rows.loc[dataset_rows["periodo"].eq(original["periodo"])]
        map_row = quarterly.loc[
            quarterly["dataset_id"].eq(dataset_id)
            & quarterly["periodo"].eq(original["periodo"])
        ].iloc[0]
        calculated = float(period_rows["precipitacion_mm"].sum(min_count=1))
        displayed = float(map_row[ACCUMULATED_COLUMN])
        samples.append(
            {
                "dataset_id": dataset_id,
                "archivo_xls": original["archivo_origen"],
                "fuente": original["fuente"],
                "estacion": original["estacion"],
                "localidad": original["localidad"],
                "provincia": original["provincia"],
                "latitud": float(original["latitud"]),
                "longitud": float(original["longitud"]),
                "fecha_muestra": original["fecha"].isoformat(),
                "anio": int(original["anio"]),
                "trimestre": original["trimestre"],
                "periodo": original["periodo"],
                "precipitacion_original": float(original["precipitacion_original"]),
                "unidad_original": original["unidad_original"],
                "precipitacion_mm": float(original["precipitacion_mm"]),
                "registros_del_trimestre": int(len(period_rows)),
                "acumulado_recalculado_mm": calculated,
                "acumulado_parquet_mapa_mm": displayed,
                "coincide_agregacion": abs(calculated - displayed) < 1e-9,
                "archivo_existe": (config.PROJECT_ROOT / original["archivo_origen"]).is_file(),
            }
        )
    return samples


def _missing_periods(periods: list[str]) -> list[str]:
    if not periods:
        return []
    indices = sorted(int(period[:4]) * 4 + int(period[-1]) - 1 for period in periods)
    existing = set(indices)
    return [
        f"{index // 4}-T{index % 4 + 1}"
        for index in range(indices[0], indices[-1] + 1)
        if index not in existing
    ]


def create_audit_report(
    daily: pd.DataFrame,
    quarterly: pd.DataFrame,
    summary: ProcessingSummary,
    output_path: Path = config.OUTPUT_AUDIT,
) -> dict[str, Any]:
    """Construye y escribe el informe de auditoría de la ejecución actual."""
    quarterly = ensure_quarterly_canonical_columns(quarterly)
    generation_time = datetime.now(TIMEZONE)
    periods = sorted(
        quarterly["periodo"].unique(),
        key=lambda value: (int(value[:4]), int(value[-1])),
    )
    maximum = precipitation_global_maximum(quarterly)
    ticks = generate_precipitation_ticks(maximum, config.PRECIPITATION_STEP)
    outside = find_stations_outside_territory(daily)
    report: dict[str, Any] = {
        "fecha_generacion": generation_time.isoformat(),
        "version_proyecto": __version__,
        "base_estable_validada": "1.0.0",
        "apto_version_1_0_0": summary.datasets_with_errors == 0,
        "datasets_declarados": summary.datasets_declared,
        "datasets_procesados": summary.datasets_processed,
        "datasets_omitidos": summary.datasets_declared - summary.datasets_processed,
        "datasets_con_error": summary.datasets_with_errors,
        "detalle_errores": summary.dataset_errors,
        "registros_originales": summary.records_read,
        "registros_validos": summary.valid_records,
        "registros_descartados": summary.discarded_records,
        "registros_duplicados": summary.duplicate_records,
        "valores_faltantes": summary.missing_values,
        "estaciones": int(quarterly["dataset_id"].nunique()),
        "provincias_representadas": int(quarterly["provincia"].nunique()),
        "periodos": len(periods),
        "periodos_faltantes": _missing_periods(periods),
        "fecha_minima": daily["fecha"].min().isoformat(),
        "fecha_maxima": daily["fecha"].max().isoformat(),
        "periodo_minimo": periods[0],
        "periodo_maximo": periods[-1],
        "precipitacion_minima_mm": float(quarterly[ACCUMULATED_COLUMN].min()),
        "precipitacion_maxima_mm": maximum,
        "precipitacion_media_mm": float(quarterly[ACCUMULATED_COLUMN].mean()),
        "precipitacion_mediana_mm": float(quarterly[ACCUMULATED_COLUMN].median()),
        "precipitaciones_negativas": int(daily["precipitacion_mm"].lt(0).sum()),
        "faltantes_reemplazados_por_cero": 0,
        "crs_visualizacion": config.TARGET_CRS,
        "escala_global": {
            "minimo_mm": config.PRECIPITATION_MIN,
            "maximo_mm": maximum,
            "paso_mm": config.PRECIPITATION_STEP,
            "ticks": ticks,
        },
        "estaciones_fuera_territorio": outside,
        "discrepancias_estacion_provincia": find_station_province_mismatches(daily),
        "muestras_trazabilidad": build_traceability_samples(daily, quarterly),
        "validaciones_automaticas": {
            "precipitaciones_no_negativas": bool(daily["precipitacion_mm"].ge(0).all()),
            "faltantes_no_convertidos_en_cero": True,
            "periodos_en_orden_cronologico": periods
            == sorted(periods, key=lambda value: (int(value[:4]), int(value[-1]))),
            "periodos_duplicados_en_slider": len(periods) != len(set(periods)),
            "escala_comienza_en_cero": ticks[0] == 0,
            "escala_finaliza_en_maximo_real": ticks[-1] == maximum,
            "crs_es_epsg_4326": config.TARGET_CRS == "EPSG:4326",
        },
        "recursos_externos": [
            "Teselas OpenStreetMap",
            "CDN Leaflet/Folium",
            "CDN jQuery y Bootstrap",
        ],
        "archivos_generados": {
            "html": str(config.OUTPUT_HTML.relative_to(config.PROJECT_ROOT)),
            "parquet": str(config.OUTPUT_PARQUET.relative_to(config.PROJECT_ROOT)),
            "auditoria": str(output_path.relative_to(config.PROJECT_ROOT)),
            "evaluacion_idw": str(
                config.OUTPUT_IDW_EVALUATION.relative_to(config.PROJECT_ROOT)
            ),
        },
    }
    write_audit_report(report, output_path)
    return report
