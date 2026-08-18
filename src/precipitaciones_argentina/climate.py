"""Evaluación y productos climáticos avanzados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .spatial import cross_validate_idw


def evaluate_idw(
    frame: pd.DataFrame, output_path: Path = config.OUTPUT_IDW_EVALUATION
) -> dict[str, Any]:
    """Calcula validación cruzada IDW por período y métricas globales ponderadas."""
    periods: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    for period, rows in frame.groupby("periodo", sort=False):
        rows = rows.loc[rows["provincia"].str.casefold().ne("sin asignar")]
        result = cross_validate_idw(
            rows["longitud"].to_numpy(float),
            rows["latitud"].to_numpy(float),
            rows["precipitacion_mm"].to_numpy(float),
            config.IDW_POWER,
        )
        if not result.sample_count:
            continue
        periods.append(
            {
                "periodo": str(period),
                "mae_mm": round(result.mae, 6),
                "rmse_mm": round(result.rmse, 6),
                "muestras": result.sample_count,
            }
        )
        absolute_errors.extend([result.mae] * result.sample_count)
        squared_errors.extend([result.rmse**2] * result.sample_count)
    report = {
        "metodo": "leave-one-station-out",
        "interpolacion": config.INTERPOLATION_METHOD,
        "potencia_idw": config.IDW_POWER,
        "mae_global_mm": float(np.mean(absolute_errors)) if absolute_errors else None,
        "rmse_global_mm": float(np.sqrt(np.mean(squared_errors))) if squared_errors else None,
        "muestras": len(absolute_errors),
        "periodos_evaluados": len(periods),
        "por_periodo": periods,
        "advertencia": "Las métricas describen error de validación, no probabilidades.",
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
