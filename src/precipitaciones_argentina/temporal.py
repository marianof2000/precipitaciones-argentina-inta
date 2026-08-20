"""Dimensión temporal y agregaciones trimestrales."""

import pandas as pd

ACCUMULATED_COLUMN = "precipitacion_acumulada_trimestral_mm"
MEAN_COLUMN = "precipitacion_promedio_trimestral_mm"


def ensure_quarterly_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Garantiza nombres canónicos sin transformar los valores científicos."""
    result = frame.copy()
    if ACCUMULATED_COLUMN not in result:
        source = "precipitacion_acumulada_mm"
        if source not in result:
            source = "precipitacion_mm"
        result[ACCUMULATED_COLUMN] = result[source]
    if MEAN_COLUMN not in result:
        if "precipitacion_media_mm" in result:
            result[MEAN_COLUMN] = result["precipitacion_media_mm"]
        elif "cantidad_observaciones" in result:
            result[MEAN_COLUMN] = result[ACCUMULATED_COLUMN].div(
                result["cantidad_observaciones"].replace(0, pd.NA)
            )
        else:
            result[MEAN_COLUMN] = result[ACCUMULATED_COLUMN]
    return result


def station_quarter_statistics(
    data: pd.DataFrame, station: str, period: str
) -> dict[str, float | int | None]:
    """Resume observaciones reales válidas de una estación y un trimestre."""
    values = pd.to_numeric(
        data.loc[
            data["estacion"].eq(station) & data["periodo"].eq(period),
            "precipitacion_mm",
        ],
        errors="coerce",
    ).dropna()
    if values.empty:
        return {"mean": None, "sum": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(values.mean()),
        "sum": float(values.sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.count()),
    }


def year_quarter(date: pd.Timestamp) -> tuple[int, str, str]:
    """Devuelve año, trimestre y período canónico."""
    timestamp = pd.Timestamp(date)
    quarter = f"T{timestamp.quarter}"
    return timestamp.year, quarter, f"{timestamp.year}-{quarter}"


def annual_quarter_values(
    frame: pd.DataFrame, dataset_id: str, selected_period: str
) -> list[dict[str, str | float | None]]:
    """Prepara T1–T4 del año seleccionado, conservando períodos sin datos."""
    try:
        year_text, selected_quarter = selected_period.split("-")
        year = int(year_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Período inválido: {selected_period}") from exc
    if selected_quarter not in {"T1", "T2", "T3", "T4"}:
        raise ValueError(f"Período inválido: {selected_period}")
    frame = ensure_quarterly_canonical_columns(frame)
    station_rows = frame.loc[frame["dataset_id"].eq(dataset_id)]
    values = station_rows.set_index("periodo")[ACCUMULATED_COLUMN]
    result: list[dict[str, str | float | None]] = []
    for quarter in ("T1", "T2", "T3", "T4"):
        period = f"{year}-{quarter}"
        value = values.get(period)
        result.append({
            "quarter": quarter,
            "period": period,
            "value": None if value is None or pd.isna(value) else float(value),
        })
    return result


def aggregate_quarterly(frame: pd.DataFrame, method: str = "sum") -> pd.DataFrame:
    """Agrega por dataset y trimestre; la suma exige al menos una observación."""
    if "id_estacion" not in frame:
        frame = frame.assign(id_estacion=frame["dataset_id"])
    if "tipo_precipitacion" not in frame:
        frame = frame.assign(tipo_precipitacion="incremental")
    if method != "sum":
        raise ValueError(
            "La variable canónica es precipitación acumulada trimestral; "
            "el único método admitido es 'sum'"
        )
    if "tipo_precipitacion" in frame and frame["tipo_precipitacion"].eq("acumulada").any():
        raise ValueError(
            "Los datasets de precipitación acumulada requieren una política explícita "
            "de intervalos; no se sumarán automáticamente"
        )
    keys = [
        "id_estacion", "dataset_id", "archivo_origen", "fuente", "estacion", "localidad",
        "provincia",
        "latitud", "longitud", "anio", "trimestre", "periodo", "unidad_original",
        "tipo_precipitacion",
    ]
    grouped = frame.groupby(keys, as_index=False, sort=False, dropna=False)
    result = grouped.agg(
        precipitacion_original=(
            "precipitacion_original",
            lambda values: values.sum(min_count=1),
        ),
        precipitacion_mm=("precipitacion_mm", lambda values: values.sum(min_count=1)),
        cantidad_observaciones=("precipitacion_mm", "count"),
        precipitacion_media_mm=("precipitacion_mm", "mean"),
        precipitacion_minima_mm=("precipitacion_mm", "min"),
        precipitacion_maxima_mm=("precipitacion_mm", "max"),
        precipitacion_acumulada_mm=(
            "precipitacion_mm", lambda values: values.sum(min_count=1)
        ),
    )
    result[ACCUMULATED_COLUMN] = result["precipitacion_acumulada_mm"]
    result[MEAN_COLUMN] = result["precipitacion_media_mm"]
    result["cantidad_registros_validos"] = result["cantidad_observaciones"]
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
    frame = ensure_quarterly_canonical_columns(frame)
    reference = frame.loc[frame["anio"].between(start_year, end_year)]
    climate = reference.groupby(["dataset_id", "trimestre"], as_index=False).agg(
        precipitacion_historica_mm=(ACCUMULATED_COLUMN, "mean"),
        anios_historicos=("anio", "nunique"),
    )
    climate.loc[
        climate["anios_historicos"].lt(minimum_years), "precipitacion_historica_mm"
    ] = float("nan")
    result = frame.merge(climate, on=["dataset_id", "trimestre"], how="left")
    result["anomalia_absoluta_mm"] = (
        result[ACCUMULATED_COLUMN] - result["precipitacion_historica_mm"]
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
