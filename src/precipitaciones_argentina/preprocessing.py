"""Normalización de observaciones heterogéneas."""

from __future__ import annotations

import logging

import pandas as pd

from .catalog import DatasetConfig
from .validation import SUPPORTED_UNITS, invalid_observation_mask

LOGGER = logging.getLogger(__name__)
NORMALIZED_COLUMNS = [
    "dataset_id", "archivo_origen", "fuente", "estacion", "localidad", "provincia",
    "latitud", "longitud", "fecha", "anio", "trimestre", "periodo",
    "precipitacion_original", "unidad_original", "tipo_precipitacion",
    "precipitacion_mm",
]


def precipitation_to_mm(values: pd.Series, unit: str) -> pd.Series:
    """Convierte precipitación a milímetros conservando faltantes."""
    normalized_unit = unit.strip().lower()
    factors = {"mm": 1.0, "cm": 10.0, "in": 25.4}
    if normalized_unit not in SUPPORTED_UNITS:
        raise ValueError(f"Unidad de precipitación desconocida: {unit}")
    return pd.to_numeric(values, errors="coerce") * factors[normalized_unit]


def normalize_dataset(
    raw: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mapea un dataset al modelo común y descarta registros inválidos trazablemente."""
    date_column = config.campos["fecha"]
    precipitation_column = config.campos["precipitacion"]
    original = pd.to_numeric(raw[precipitation_column], errors="coerce")
    frame = pd.DataFrame(
        {
            "dataset_id": config.dataset_id,
            "archivo_origen": config.archivo,
            "fuente": config.fuente,
            "estacion": config.estacion,
            "localidad": config.localidad,
            "provincia": config.provincia,
            "latitud": config.latitud,
            "longitud": config.longitud,
            "fecha": pd.to_datetime(raw[date_column], errors="coerce"),
            "precipitacion_original": original,
            "unidad_original": config.unidad_precipitacion,
            "tipo_precipitacion": config.tipo_precipitacion,
            "precipitacion_mm": precipitation_to_mm(original, config.unidad_precipitacion),
        }
    )
    missing = int(frame[["fecha", "precipitacion_original"]].isna().any(axis=1).sum())
    duplicates = frame.duplicated(subset=["dataset_id", "fecha"], keep="first")
    invalid = invalid_observation_mask(frame)
    negative = int(frame["precipitacion_mm"].lt(0).sum())
    if negative:
        LOGGER.warning("%s: %d precipitaciones negativas", config.dataset_id, negative)
    valid = frame.loc[~invalid & ~duplicates].copy()
    valid["anio"] = valid["fecha"].dt.year.astype("int32")
    valid["trimestre"] = (valid["fecha"].dt.quarter).map(lambda value: f"T{value}")
    valid["periodo"] = valid["anio"].astype("string").str.cat(
        valid["trimestre"].astype("string"), sep="-"
    )
    return valid[NORMALIZED_COLUMNS], {
        "read": len(frame),
        "valid": len(valid),
        "discarded": int((invalid | duplicates).sum()),
        "duplicates": int(duplicates.sum()),
        "missing": missing,
    }
