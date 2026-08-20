"""Lectura del catálogo geográfico y del manifiesto del CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class CatalogError(ValueError):
    """Indica que un archivo de metadatos no cumple su contrato."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"No se pudo leer {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogError(f"{path.name} debe contener un objeto JSON")
    return payload


def load_station_catalog(path: Path) -> pd.DataFrame:
    """Carga ``estaciones.json`` exclusivamente como catálogo geográfico."""
    payload = _read_json(path)
    entries = payload.get("estaciones")
    if not isinstance(entries, list):
        raise CatalogError("El catálogo debe contener una lista 'estaciones'")
    source = str(payload.get("configuracion", {}).get("fuente_nombre", payload.get("fuente", "")))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metadata = entry.get("metadata_origen", {})
        station_id = str(entry.get("id_estacion", "")).strip()
        if not station_id:
            raise CatalogError("Una estación no tiene id_estacion")
        try:
            latitude = float(metadata["latitud"])
            longitude = float(metadata["longitud"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError(f"Coordenadas inválidas para {station_id}") from exc
        rows.append({
            "id_estacion": station_id, "dataset_id": station_id,
            "estacion": str(metadata.get("nombre", station_id)),
            "localidad": str(metadata.get("localidad", "")),
            "provincia": str(metadata.get("provincia", "")),
            "latitud": latitude, "longitud": longitude, "fuente": source,
        })
    frame = pd.DataFrame(rows)
    if frame["id_estacion"].duplicated().any():
        duplicates = frame.loc[frame["id_estacion"].duplicated(), "id_estacion"].tolist()
        raise CatalogError(f"El catálogo contiene identificadores duplicados: {duplicates}")
    declared = payload.get("cantidad_estaciones")
    if declared is not None and int(declared) != len(frame):
        raise CatalogError(f"cantidad_estaciones={declared} no coincide con {len(frame)} entradas")
    return frame


def load_csv_manifest(path: Path) -> dict[str, Any]:
    """Carga y valida los campos estructurales de ``estaciones_csv.json``."""
    payload = _read_json(path)
    required = {"cantidad_registros", "cantidad_columnas", "columnas", "tamano_bytes", "sha256"}
    missing = sorted(required.difference(payload))
    if missing:
        raise CatalogError(f"Faltan campos en el manifiesto: {', '.join(missing)}")
    if not isinstance(payload["columnas"], list):
        raise CatalogError("manifest.columnas debe ser una lista")
    if int(payload["cantidad_columnas"]) != len(payload["columnas"]):
        raise CatalogError("cantidad_columnas no coincide con la lista columnas")
    return payload
