"""Carga y verificación de las observaciones consolidadas en CSV."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetLoadError(RuntimeError):
    """Indica que el CSV no puede utilizarse de forma segura."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_observations_csv(
    path: Path, manifest: dict[str, Any], *, verify_sha256: bool = True
) -> pd.DataFrame:
    """Lee el CSV y contrasta contenido y metadatos con su manifiesto."""
    if not path.is_file():
        raise DatasetLoadError(f"No se encontró {path}")
    actual_size = path.stat().st_size
    expected_size = int(manifest["tamano_bytes"])
    if actual_size != expected_size:
        raise DatasetLoadError(f"Tamaño CSV inesperado: {actual_size}; manifiesto: {expected_size}")
    if verify_sha256:
        actual_hash = file_sha256(path)
        expected_hash = str(manifest["sha256"]).lower()
        if actual_hash != expected_hash:
            raise DatasetLoadError(
                f"SHA-256 CSV no coincide: {actual_hash}; manifiesto: {expected_hash}"
            )
    try:
        frame = pd.read_csv(path, dtype={"id_estacion": "string"})
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DatasetLoadError(f"No se pudo leer {path}: {exc}") from exc
    expected_columns = [str(value) for value in manifest["columnas"]]
    if frame.columns.tolist() != expected_columns:
        raise DatasetLoadError("Las columnas del CSV no coinciden con el manifiesto")
    expected_rows = int(manifest["cantidad_registros"])
    if len(frame) != expected_rows:
        raise DatasetLoadError(f"Filas CSV inesperadas: {len(frame)}; manifiesto: {expected_rows}")
    if len(frame.columns) != int(manifest["cantidad_columnas"]):
        raise DatasetLoadError("La cantidad de columnas del CSV no coincide con el manifiesto")
    return frame
