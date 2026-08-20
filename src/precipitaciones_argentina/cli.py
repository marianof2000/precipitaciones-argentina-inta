"""Orquestación del pipeline CSV consolidado."""

from __future__ import annotations

import logging

import pandas as pd

from . import config
from .audit import create_audit_report, write_audit_report
from .catalog import load_csv_manifest, load_station_catalog
from .climate import evaluate_idw
from .loaders import read_observations_csv
from .preprocessing import normalize_observations
from .temporal import add_climate_anomalies, aggregate_quarterly
from .validation import ProcessingSummary
from .visualization import generate_map

LOGGER = logging.getLogger(__name__)


def run_pipeline() -> tuple[pd.DataFrame, ProcessingSummary]:
    """Procesa el CSV único, lo cruza con el catálogo y escribe las salidas."""
    LOGGER.info("Catálogo geográfico: %s", config.STATION_CATALOG_FILE)
    stations = load_station_catalog(config.STATION_CATALOG_FILE)
    manifest = load_csv_manifest(config.OBSERVATIONS_MANIFEST)
    raw = read_observations_csv(
        config.OBSERVATIONS_CSV, manifest, verify_sha256=config.VERIFY_CSV_SHA256
    )
    daily, metrics = normalize_observations(
        raw, stations, source_file=str(config.OBSERVATIONS_CSV.relative_to(config.PROJECT_ROOT))
    )
    differences = list(manifest.get("diferencias_estructurales", []))
    manifest_catalog = manifest.get("cantidad_estaciones_catalogo")
    if manifest_catalog is not None and int(manifest_catalog) != len(stations):
        differences.append(
            f"Manifiesto declara {manifest_catalog} estaciones; "
            f"catálogo actual contiene {len(stations)}"
        )
    summary = ProcessingSummary(
        datasets_declared=1, datasets_processed=1, records_read=int(metrics["read"]),
        valid_records=int(metrics["valid"]), discarded_records=int(metrics["discarded"]),
        duplicate_records=int(metrics["duplicates"]), missing_values=int(metrics["missing"]),
        source_file=str(config.OBSERVATIONS_CSV.relative_to(config.PROJECT_ROOT)),
        stations_catalog=len(stations), stations_observed=int(daily["id_estacion"].nunique()),
        unknown_station_ids=list(metrics["unknown_station_ids"]), manifest_differences=differences,
    )
    for difference in differences:
        LOGGER.warning("Diferencia de manifiesto: %s", difference)
    quarterly = aggregate_quarterly(daily, config.AGGREGATION_METHOD)
    quarterly = add_climate_anomalies(
        quarterly, start_year=config.CLIMATOLOGY_START_YEAR,
        end_year=config.CLIMATOLOGY_END_YEAR, minimum_years=config.MIN_HISTORICAL_YEARS,
    )
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
        html_path = generate_map(quarterly, audit=audit, station_catalog=stations)
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
        "Resumen — fuentes declaradas: %d; procesadas: %d; con errores: %d; "
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
