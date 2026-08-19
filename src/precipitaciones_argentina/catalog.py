"""Lectura y validación del catálogo maestro de estaciones."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CatalogError(ValueError):
    """Indica que el catálogo no cumple su contrato."""


@dataclass(frozen=True)
class DatasetConfig:
    """Configuración resuelta de un dataset declarado."""

    dataset_id: str
    archivo: str
    fuente: str
    estacion: str
    localidad: str
    provincia: str
    latitud: float
    longitud: float
    hoja: str | int = 0
    unidad_precipitacion: str = "mm"
    campos: dict[str, str] = field(default_factory=dict)
    preferir_coordenadas_catalogo: bool = True
    tipo_precipitacion: str = "incremental"
    extras: dict[str, Any] = field(default_factory=dict)


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise CatalogError(f"Falta '{key}' en {context}")
    return value


def _precipitation_type(mapping: dict[str, Any]) -> str:
    value = str(mapping.get("tipo_precipitacion", "incremental"))
    if value not in {"incremental", "acumulada"}:
        raise CatalogError("tipo_precipitacion debe ser 'incremental' o 'acumulada'")
    return value


def _from_inta(entry: dict[str, Any], defaults: dict[str, Any]) -> DatasetConfig:
    metadata = _required(entry, "metadata_origen", "estación")
    download = _required(entry, "descarga", "estación")
    fields = defaults.get("campos", {})
    if not {"fecha", "precipitacion"}.issubset(fields):
        raise CatalogError("configuracion.campos debe declarar fecha y precipitacion")
    return DatasetConfig(
        dataset_id=str(_required(entry, "id_estacion", "estación")),
        archivo=str(_required(download, "archivo", "descarga")),
        fuente=str(defaults.get("fuente_nombre", "INTA")),
        estacion=str(_required(metadata, "nombre", "metadata_origen")),
        localidad=str(metadata.get("localidad", "")),
        provincia=str(metadata.get("provincia", "")),
        latitud=float(_required(metadata, "latitud", "metadata_origen")),
        longitud=float(_required(metadata, "longitud", "metadata_origen")),
        hoja=defaults.get("hoja", 0),
        unidad_precipitacion=str(defaults.get("unidad_precipitacion", "mm")),
        tipo_precipitacion=_precipitation_type({**defaults, **entry}),
        campos=dict(fields),
        preferir_coordenadas_catalogo=bool(
            defaults.get("preferir_coordenadas_catalogo", True)
        ),
        extras={"metadata_origen": metadata, "descarga": download},
    )


def _from_generic(entry: dict[str, Any], defaults: dict[str, Any]) -> DatasetConfig:
    merged = defaults | entry
    fields = merged.get("campos", {})
    if not {"fecha", "precipitacion"}.issubset(fields):
        raise CatalogError("campos debe declarar fecha y precipitacion")
    known = {
        "id", "archivo", "fuente", "estacion", "localidad", "provincia", "latitud",
        "longitud", "hoja", "unidad_precipitacion", "tipo_precipitacion", "campos",
        "preferir_coordenadas_catalogo",
    }
    return DatasetConfig(
        dataset_id=str(_required(merged, "id", "dataset")),
        archivo=str(_required(merged, "archivo", "dataset")),
        fuente=str(_required(merged, "fuente", "dataset")),
        estacion=str(_required(merged, "estacion", "dataset")),
        localidad=str(merged.get("localidad", "")),
        provincia=str(merged.get("provincia", "")),
        latitud=float(_required(merged, "latitud", "dataset")),
        longitud=float(_required(merged, "longitud", "dataset")),
        hoja=merged.get("hoja", 0),
        unidad_precipitacion=str(merged.get("unidad_precipitacion", "mm")),
        tipo_precipitacion=_precipitation_type(merged),
        campos=dict(fields),
        preferir_coordenadas_catalogo=bool(
            merged.get("preferir_coordenadas_catalogo", True)
        ),
        extras={key: value for key, value in entry.items() if key not in known},
    )


def load_catalog(path: Path) -> list[DatasetConfig]:
    """Carga formatos genérico e INTA sin explorar archivos fuera del catálogo."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"No se pudo leer el catálogo {path}: {exc}") from exc
    defaults = payload.get("configuracion", {})
    entries = payload.get("datasets", payload.get("estaciones"))
    if not isinstance(entries, list):
        raise CatalogError("El catálogo debe contener una lista datasets o estaciones")
    factory = _from_generic if "datasets" in payload else _from_inta
    datasets = [factory(entry, defaults) for entry in entries]
    identifiers = [item.dataset_id for item in datasets]
    if len(identifiers) != len(set(identifiers)):
        raise CatalogError("El catálogo contiene identificadores duplicados")
    return datasets
