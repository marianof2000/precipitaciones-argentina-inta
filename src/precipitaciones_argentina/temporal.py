"""Dimensión temporal y agregaciones trimestrales."""

import pandas as pd


def year_quarter(date: pd.Timestamp) -> tuple[int, str, str]:
    """Devuelve año, trimestre y período canónico."""
    timestamp = pd.Timestamp(date)
    quarter = f"T{timestamp.quarter}"
    return timestamp.year, quarter, f"{timestamp.year}-{quarter}"


def aggregate_quarterly(frame: pd.DataFrame, method: str = "sum") -> pd.DataFrame:
    """Agrega por dataset y trimestre; la suma exige al menos una observación."""
    supported = {"sum", "mean", "median", "min", "max"}
    if method not in supported:
        raise ValueError(f"Agregación no soportada: {method}")
    keys = [
        "dataset_id", "archivo_origen", "fuente", "estacion", "localidad", "provincia",
        "latitud", "longitud", "anio", "trimestre", "periodo", "unidad_original",
    ]
    grouped = frame.groupby(keys, as_index=False, sort=False, dropna=False)
    if method == "sum":
        result = grouped.agg(
            precipitacion_original=(
                "precipitacion_original",
                lambda values: values.sum(min_count=1),
            ),
            precipitacion_mm=("precipitacion_mm", lambda values: values.sum(min_count=1)),
            cantidad_observaciones=("precipitacion_mm", "count"),
        )
    else:
        result = grouped.agg(
            precipitacion_original=("precipitacion_original", method),
            precipitacion_mm=("precipitacion_mm", method),
            cantidad_observaciones=("precipitacion_mm", "count"),
        )
    quarter_number = result["trimestre"].str.removeprefix("T").astype(int)
    return result.assign(_quarter=quarter_number).sort_values(
        ["anio", "_quarter", "dataset_id"]
    ).drop(columns="_quarter").reset_index(drop=True)
