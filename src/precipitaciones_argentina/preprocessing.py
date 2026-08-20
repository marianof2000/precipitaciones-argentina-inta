"""Normalización de observaciones y unión con el catálogo geográfico."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRED_OBSERVATION_COLUMNS = {"id_estacion", "fecha", "precipitacion_pluviometrica"}
NORMALIZED_COLUMNS = [
    "id_estacion", "dataset_id", "archivo_origen", "fuente", "estacion", "localidad",
    "provincia", "latitud", "longitud", "fecha", "anio", "trimestre", "periodo",
    "precipitacion_original", "unidad_original", "tipo_precipitacion", "precipitacion_mm",
]


def normalize_observations(
    raw: pd.DataFrame, stations: pd.DataFrame, *, source_file: str = "datos/estaciones.csv"
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normaliza el CSV y hace un join many-to-one exclusivamente por ``id_estacion``."""
    missing_columns = sorted(REQUIRED_OBSERVATION_COLUMNS.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing_columns)}")
    observations = raw[["id_estacion", "fecha", "precipitacion_pluviometrica"]].copy()
    observations["id_estacion"] = observations["id_estacion"].astype("string").str.strip()
    observations["fecha"] = pd.to_datetime(observations["fecha"], errors="coerce")
    observations["precipitacion_original"] = pd.to_numeric(
        observations.pop("precipitacion_pluviometrica"), errors="coerce"
    )
    catalog_ids = set(stations["id_estacion"].astype(str))
    unknown_ids = sorted(set(observations["id_estacion"].dropna().astype(str)) - catalog_ids)
    if unknown_ids:
        LOGGER.warning(
            "%d estaciones del CSV no están en el catálogo y serán omitidas: %s",
            len(unknown_ids), ", ".join(unknown_ids),
        )
    frame = observations.merge(
        stations, on="id_estacion", how="left", validate="many_to_one", indicator=True
    )
    unmatched_rows = int(frame["_merge"].ne("both").sum())
    frame = frame.loc[frame["_merge"].eq("both")].drop(columns="_merge")
    frame["dataset_id"] = frame["id_estacion"]
    frame["archivo_origen"] = source_file
    frame["unidad_original"] = "mm"
    frame["tipo_precipitacion"] = "incremental"
    frame["precipitacion_mm"] = frame["precipitacion_original"]
    duplicates = frame.duplicated(subset=["id_estacion", "fecha"], keep="first")
    missing = frame[["fecha", "precipitacion_original"]].isna().any(axis=1)
    negative = frame["precipitacion_mm"].lt(0)
    invalid_coordinates = ~frame["latitud"].between(-90, 90) | ~frame["longitud"].between(-180, 180)
    invalid = missing | negative | invalid_coordinates | duplicates
    if negative.any():
        LOGGER.warning("Se omitieron %d precipitaciones negativas", int(negative.sum()))
    valid = frame.loc[~invalid].copy()
    valid["anio"] = valid["fecha"].dt.year.astype("int32")
    valid["trimestre"] = "T" + valid["fecha"].dt.quarter.astype(str)
    valid["periodo"] = valid["anio"].astype("string") + "-" + valid["trimestre"]
    metrics: dict[str, object] = {
        "read": len(raw), "valid": len(valid), "discarded": int(invalid.sum()) + unmatched_rows,
        "duplicates": int(duplicates.sum()), "missing": int(missing.sum()),
        "negative": int(negative.sum()), "unknown_station_ids": unknown_ids,
        "unmatched_rows": unmatched_rows,
    }
    return valid[NORMALIZED_COLUMNS], metrics
