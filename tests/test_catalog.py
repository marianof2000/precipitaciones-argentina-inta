import json

import pytest

from precipitaciones_argentina.catalog import CatalogError, load_csv_manifest, load_station_catalog


def test_load_station_catalog(tmp_path):
    path = tmp_path / "estaciones.json"
    payload = {"cantidad_estaciones": 1, "configuracion": {"fuente_nombre": "INTA"},
        "estaciones": [{"id_estacion": "A1", "metadata_origen": {"nombre": "Estación",
        "localidad": "Localidad", "provincia": "Provincia", "latitud": -34, "longitud": -58}}]}
    path.write_text(json.dumps(payload))
    frame = load_station_catalog(path)
    assert frame.loc[0, "id_estacion"] == "A1"
    assert frame.loc[0, "fuente"] == "INTA"


def test_catalog_rejects_duplicate_ids(tmp_path):
    station = {"id_estacion": "A1", "metadata_origen": {"latitud": -34, "longitud": -58}}
    path = tmp_path / "estaciones.json"
    path.write_text(json.dumps({"estaciones": [station, station]}))
    with pytest.raises(CatalogError, match="duplicados"):
        load_station_catalog(path)


def test_manifest_requires_structural_fields(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    with pytest.raises(CatalogError, match="Faltan campos"):
        load_csv_manifest(path)
