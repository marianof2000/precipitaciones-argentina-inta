"""Orquestación de la Etapa 1 del pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from . import config
from .audit import create_audit_report, write_audit_report
from .catalog import load_catalog
from .climate import evaluate_idw
from .loaders import DatasetLoadError, read_excel_dataset
from .preprocessing import NORMALIZED_COLUMNS, normalize_dataset
from .temporal import add_climate_anomalies, aggregate_quarterly
from .validation import ProcessingSummary, valid_latitude, valid_longitude
from .visualization import generate_map

LOGGER = logging.getLogger(__name__)


def run_pipeline() -> tuple[pd.DataFrame, ProcessingSummary]:
    """Procesa todos y sólo los datasets declarados y escribe el Parquet trimestral."""
    LOGGER.info("Leyendo configuración: %s", config.STATIONS_FILE)
    datasets = load_catalog(config.STATIONS_FILE)
    summary = ProcessingSummary(datasets_declared=len(datasets))
    normalized_frames: list[pd.DataFrame] = []
    LOGGER.info("Datasets declarados: %d", len(datasets))

    for position, dataset in enumerate(datasets, start=1):
        LOGGER.info("[%d/%d] %s", position, len(datasets), dataset.archivo)
        if not valid_latitude(dataset.latitud) or not valid_longitude(dataset.longitud):
            LOGGER.error("%s: coordenadas configuradas inválidas", dataset.dataset_id)
            summary.datasets_with_errors += 1
            summary.dataset_errors.append(
                {"dataset_id": dataset.dataset_id, "archivo": dataset.archivo,
                 "error": "Coordenadas configuradas inválidas"}
            )
            continue
        try:
            raw = read_excel_dataset(dataset, config.DATA_DIR)
            normalized, metrics = normalize_dataset(raw, dataset)
        except (DatasetLoadError, ValueError, TypeError) as exc:
            LOGGER.error("%s: %s", dataset.dataset_id, exc)
            summary.datasets_with_errors += 1
            summary.dataset_errors.append(
                {"dataset_id": dataset.dataset_id, "archivo": dataset.archivo, "error": str(exc)}
            )
            continue
        normalized_frames.append(normalized)
        summary.datasets_processed += 1
        summary.records_read += metrics["read"]
        summary.valid_records += metrics["valid"]
        summary.discarded_records += metrics["discarded"]
        summary.duplicate_records += metrics["duplicates"]
        summary.missing_values += metrics["missing"]
        LOGGER.info(
            "Registros leídos: %d; válidos: %d; descartados: %d",
            metrics["read"], metrics["valid"], metrics["discarded"],
        )

    if normalized_frames:
        daily = pd.concat(normalized_frames, ignore_index=True)
        quarterly = aggregate_quarterly(daily, config.AGGREGATION_METHOD)
        quarterly = add_climate_anomalies(
            quarterly,
            start_year=config.CLIMATOLOGY_START_YEAR,
            end_year=config.CLIMATOLOGY_END_YEAR,
            minimum_years=config.MIN_HISTORICAL_YEARS,
        )
    else:
        quarterly = pd.DataFrame(columns=[*NORMALIZED_COLUMNS, "cantidad_observaciones"])
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quarterly.to_parquet(config.OUTPUT_PARQUET, index=False)
    LOGGER.info("Parquet generado correctamente: %s", config.OUTPUT_PARQUET)
    if not quarterly.empty:
        audit = create_audit_report(daily, quarterly, summary)
        evaluation = evaluate_idw(quarterly)
        audit["evaluacion_idw"] = {
            key: value for key, value in evaluation.items() if key != "por_periodo"
        }
        LOGGER.info("Auditoría generada correctamente: %s", config.OUTPUT_AUDIT)
        LOGGER.info("Generando mapa temporal...")
        html_path = generate_map(quarterly, audit=audit)
        audit["tamano_html_bytes"] = html_path.stat().st_size
        audit["tamano_html_mb"] = round(html_path.stat().st_size / 1024**2, 3)
        write_audit_report(audit)
        LOGGER.info(
            "HTML generado correctamente: %s (%d bytes; %.3f MB)",
            html_path, html_path.stat().st_size, html_path.stat().st_size / 1024**2,
        )
    _log_summary(summary)
    return quarterly, summary


def _log_summary(summary: ProcessingSummary) -> None:
    LOGGER.info(
        "Resumen — declarados: %d; procesados: %d; con errores: %d; "
        "leídos: %d; válidos: %d; descartados: %d; duplicados: %d; faltantes: %d",
        summary.datasets_declared, summary.datasets_processed, summary.datasets_with_errors,
        summary.records_read, summary.valid_records, summary.discarded_records,
        summary.duplicate_records, summary.missing_values,
    )


def main() -> None:
    """Punto de entrada del comando ``precipitaciones``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()
