import pandas as pd
import pytest

from precipitaciones_argentina.catalog import DatasetConfig
from precipitaciones_argentina.loaders import DatasetLoadError, read_excel_dataset


def config(filename="sample.xlsx", sheet="Datos", precipitation="Lluvia"):
    return DatasetConfig(
        "x", filename, "F", "E", "L", "P", -34, -58, sheet, "mm",
        {"fecha": "Fecha", "precipitacion": precipitation},
    )


def test_existing_xlsx_and_sheet(tmp_path):
    pd.DataFrame({"Fecha": ["2024-01-01"], "Lluvia": [2]}).to_excel(
        tmp_path / "sample.xlsx", sheet_name="Datos", index=False
    )
    assert len(read_excel_dataset(config(), tmp_path)) == 1


def test_missing_dataset(tmp_path):
    with pytest.raises(DatasetLoadError, match="No se encontró"):
        read_excel_dataset(config(), tmp_path)


def test_missing_sheet(tmp_path):
    pd.DataFrame({"Fecha": [], "Lluvia": []}).to_excel(tmp_path / "sample.xlsx", index=False)
    with pytest.raises(DatasetLoadError, match="Hoja inexistente"):
        read_excel_dataset(config(), tmp_path)


def test_missing_column(tmp_path):
    pd.DataFrame({"Fecha": []}).to_excel(
        tmp_path / "sample.xlsx", sheet_name="Datos", index=False
    )
    with pytest.raises(DatasetLoadError, match="Columnas inexistentes"):
        read_excel_dataset(config(), tmp_path)
