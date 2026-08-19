"""Estadísticas reutilizables por período."""

import math

import numpy as np
import pandas as pd

from .temporal import ACCUMULATED_COLUMN, ensure_quarterly_canonical_columns


def precipitation_global_maximum(frame: pd.DataFrame) -> float:
    """Obtiene el máximo global válido o cero para un conjunto vacío."""
    maximum = ensure_quarterly_canonical_columns(frame)[ACCUMULATED_COLUMN].max()
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


def normalize_precipitation_for_color(
    value_mm: float, maximum_mm: float, scale: str = "log"
) -> float:
    """Normaliza precipitación real sólo para asignación cromática."""
    if scale not in {"linear", "log"}:
        raise ValueError("scale debe ser 'linear' o 'log'")
    if pd.isna(value_mm):
        return float("nan")
    if maximum_mm <= 0:
        return 0.0
    value = min(max(float(value_mm), 0.0), float(maximum_mm))
    if scale == "linear":
        return value / maximum_mm
    return math.log1p(value) / math.log1p(maximum_mm)


def generate_log_legend_ticks(maximum_mm: float) -> list[float]:
    """Genera marcas representativas en mm para una transformación log1p."""
    if maximum_mm < 0:
        raise ValueError("maximum_mm debe ser no negativo")
    if maximum_mm == 0:
        return [0]
    candidates = {0.0, 10.0, 25.0, 50.0}
    highest_power = max(2, int(math.ceil(math.log10(maximum_mm))))
    for power in range(2, highest_power + 1):
        base = 10**power
        candidates.update({float(base), float(2 * base), float(5 * base)})
    ticks = sorted(value for value in candidates if value < maximum_mm)
    ticks.append(float(maximum_mm))
    return [
        int(value) if float(value).is_integer() else float(value)
        for value in np.unique(ticks)
    ]


def thin_legend_ticks(ticks: list[float], maximum_count: int = 9) -> list[float]:
    """Reduce marcas visuales conservando extremos y distribución uniforme."""
    if maximum_count < 2:
        raise ValueError("maximum_count debe ser al menos 2")
    if len(ticks) <= maximum_count:
        return ticks
    indices = np.linspace(0, len(ticks) - 1, maximum_count)
    return [ticks[index] for index in np.unique(np.rint(indices).astype(int))]


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
    frame = ensure_quarterly_canonical_columns(frame)
    result = frame.groupby("periodo", as_index=False, sort=False).agg(
        cantidad_observaciones=("cantidad_observaciones", "sum"),
        cantidad_estaciones=("estacion", "nunique"),
        cantidad_datasets=("dataset_id", "nunique"),
        cantidad_fuentes=("fuente", "nunique"),
        precipitacion_minima=(ACCUMULATED_COLUMN, "min"),
        precipitacion_maxima=(ACCUMULATED_COLUMN, "max"),
        precipitacion_media=(ACCUMULATED_COLUMN, "mean"),
        precipitacion_mediana=(ACCUMULATED_COLUMN, "median"),
    )
    return result[columns]
