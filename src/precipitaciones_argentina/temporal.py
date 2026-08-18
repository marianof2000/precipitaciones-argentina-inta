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


def add_climate_anomalies(
    frame: pd.DataFrame,
    *,
    start_year: int = 1991,
    end_year: int = 2020,
    minimum_years: int = 5,
) -> pd.DataFrame:
    """Agrega normal climática y anomalías por dataset y trimestre calendario.

    La anomalía relativa queda ausente si la normal es cero. Una normal requiere años
    distintos suficientes dentro del período de referencia, evitando falsas climatologías.
    """
    if start_year > end_year or minimum_years < 1:
        raise ValueError("Período climatológico o mínimo de años inválido")
    reference = frame.loc[frame["anio"].between(start_year, end_year)]
    climate = reference.groupby(["dataset_id", "trimestre"], as_index=False).agg(
        precipitacion_historica_mm=("precipitacion_mm", "mean"),
        anios_historicos=("anio", "nunique"),
    )
    climate.loc[
        climate["anios_historicos"].lt(minimum_years), "precipitacion_historica_mm"
    ] = float("nan")
    result = frame.merge(climate, on=["dataset_id", "trimestre"], how="left")
    result["anomalia_absoluta_mm"] = (
        result["precipitacion_mm"] - result["precipitacion_historica_mm"]
    )
    valid_relative = result["precipitacion_historica_mm"].gt(0)
    result["anomalia_relativa_pct"] = float("nan")
    result.loc[valid_relative, "anomalia_relativa_pct"] = (
        result.loc[valid_relative, "anomalia_absoluta_mm"]
        / result.loc[valid_relative, "precipitacion_historica_mm"]
        * 100
    )
    return result


def station_time_series(frame: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Devuelve la serie cronológica completa de una estación."""
    return frame.loc[frame["dataset_id"].eq(dataset_id)].sort_values(
        ["anio", "trimestre"]
    ).reset_index(drop=True)


def interannual_comparison(
    frame: pd.DataFrame, dataset_id: str, quarter: str
) -> pd.DataFrame:
    """Compara un trimestre calendario de una estación entre años."""
    if quarter not in {"T1", "T2", "T3", "T4"}:
        raise ValueError(f"Trimestre inválido: {quarter}")
    return station_time_series(frame, dataset_id).loc[
        lambda rows: rows["trimestre"].eq(quarter)
    ].reset_index(drop=True)
