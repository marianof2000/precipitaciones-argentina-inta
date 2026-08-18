"""Estadísticas reutilizables por período."""

import pandas as pd


def precipitation_global_maximum(frame: pd.DataFrame) -> float:
    """Obtiene el máximo global válido o cero para un conjunto vacío."""
    maximum = frame["precipitacion_mm"].max()
    return 0.0 if pd.isna(maximum) else float(maximum)


def generate_precipitation_ticks(maximum: float, step: float = 10) -> list[float]:
    """Genera cortes regulares e incluye el máximo real sin redondearlo."""
    if maximum < 0 or step <= 0:
        raise ValueError("maximum debe ser no negativo y step positivo")
    ticks = [float(value) for value in range(0, int(maximum // step) * int(step) + 1, int(step))]
    if not ticks:
        ticks = [0.0]
    if maximum > ticks[-1]:
        ticks.append(float(maximum))
    return [int(value) if value.is_integer() else value for value in ticks]


def quarterly_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores comparables para cada período trimestral."""
    columns = [
        "periodo",
        "cantidad_observaciones",
        "cantidad_estaciones",
        "cantidad_datasets",
        "cantidad_fuentes",
        "precipitacion_minima",
        "precipitacion_maxima",
        "precipitacion_media",
        "precipitacion_mediana",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.groupby("periodo", as_index=False, sort=False).agg(
        cantidad_observaciones=("cantidad_observaciones", "sum"),
        cantidad_estaciones=("estacion", "nunique"),
        cantidad_datasets=("dataset_id", "nunique"),
        cantidad_fuentes=("fuente", "nunique"),
        precipitacion_minima=("precipitacion_mm", "min"),
        precipitacion_maxima=("precipitacion_mm", "max"),
        precipitacion_media=("precipitacion_mm", "mean"),
        precipitacion_mediana=("precipitacion_mm", "median"),
    )
    return result[columns]
