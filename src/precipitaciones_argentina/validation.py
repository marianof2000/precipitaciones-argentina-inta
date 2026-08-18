"""Reglas de validación y métricas del pipeline."""

from dataclasses import dataclass, field

import pandas as pd

SUPPORTED_UNITS = {"mm", "cm", "in"}


@dataclass
class ProcessingSummary:
    """Contadores acumulados de la ejecución."""

    datasets_declared: int = 0
    datasets_processed: int = 0
    datasets_with_errors: int = 0
    records_read: int = 0
    valid_records: int = 0
    discarded_records: int = 0
    duplicate_records: int = 0
    missing_values: int = 0
    dataset_errors: list[dict[str, str]] = field(default_factory=list)


def valid_latitude(value: float) -> bool:
    return -90 <= value <= 90


def valid_longitude(value: float) -> bool:
    return -180 <= value <= 180


def invalid_observation_mask(frame: pd.DataFrame) -> pd.Series:
    """Marca observaciones inválidas; un faltante nunca se transforma en cero."""
    return (
        frame["fecha"].isna()
        | frame["precipitacion_original"].isna()
        | frame["precipitacion_mm"].isna()
        | frame["precipitacion_mm"].lt(0)
        | ~frame["latitud"].between(-90, 90)
        | ~frame["longitud"].between(-180, 180)
    )
