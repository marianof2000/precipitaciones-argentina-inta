"""Carga controlada de libros Excel declarados en el catálogo."""

from pathlib import Path

import pandas as pd

from .catalog import DatasetConfig


class DatasetLoadError(RuntimeError):
    """Error recuperable al cargar un dataset individual."""


def resolve_dataset_path(data_dir: Path, configured_path: str) -> Path:
    """Resuelve rutas relativas al proyecto o al directorio datos."""
    path = Path(configured_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == data_dir.name:
        return data_dir.parent / path
    return data_dir / path


def read_excel_dataset(config: DatasetConfig, data_dir: Path) -> pd.DataFrame:
    """Lee un XLS/XLSX y verifica hoja y columnas sin inferir nombres."""
    path = resolve_dataset_path(data_dir, config.archivo)
    if not path.is_file():
        raise DatasetLoadError(f"No se encontró {path}")
    suffix = path.suffix.lower()
    engines = {".xls": "xlrd", ".xlsx": "openpyxl"}
    if suffix not in engines:
        raise DatasetLoadError(f"Formato Excel no soportado: {path.name}")
    try:
        book = pd.ExcelFile(path, engine=engines[suffix])
    except (OSError, ValueError) as exc:
        raise DatasetLoadError(f"No se pudo abrir {path}: {exc}") from exc
    sheet = config.hoja
    if isinstance(sheet, str) and sheet not in book.sheet_names:
        raise DatasetLoadError(f"Hoja inexistente '{sheet}' en {path.name}")
    if isinstance(sheet, int) and not 0 <= sheet < len(book.sheet_names):
        raise DatasetLoadError(f"Índice de hoja inexistente {sheet} en {path.name}")
    try:
        frame = pd.read_excel(book, sheet_name=sheet)
    except (ValueError, TypeError) as exc:
        raise DatasetLoadError(f"Error leyendo {path.name}: {exc}") from exc
    required = set(config.campos.values())
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetLoadError(
            f"Columnas inexistentes en {path.name}: {', '.join(sorted(missing))}"
        )
    return frame

