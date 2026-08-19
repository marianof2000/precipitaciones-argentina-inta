import json

import pytest

from precipitaciones_argentina.catalog import CatalogError, load_catalog


def test_load_generic_catalog(tmp_path):
    path = tmp_path / "estaciones.json"
    path.write_text(json.dumps({"datasets": [{
        "id": "x", "archivo": "x.xls", "fuente": "fuente", "estacion": "E",
        "localidad": "L", "provincia": "P", "latitud": -34, "longitud": -58,
        "campos": {"fecha": "Fecha", "precipitacion": "Lluvia"},
    }]}))
    assert load_catalog(path)[0].dataset_id == "x"
    assert load_catalog(path)[0].tipo_precipitacion == "incremental"


def test_catalog_rejects_unknown_precipitation_semantics(tmp_path):
    path = tmp_path / "estaciones.json"
    path.write_text(json.dumps({"datasets": [{
        "id": "x", "archivo": "x.xls", "fuente": "F", "estacion": "E",
        "latitud": -34, "longitud": -58, "tipo_precipitacion": "inferida",
        "campos": {"fecha": "Fecha", "precipitacion": "Lluvia"},
    }]}))
    with pytest.raises(CatalogError, match="tipo_precipitacion"):
        load_catalog(path)


def test_inta_station_can_override_default_precipitation_type(tmp_path):
    path = tmp_path / "estaciones.json"
    path.write_text(json.dumps({
        "configuracion": {
            "tipo_precipitacion": "incremental",
            "campos": {"fecha": "Fecha", "precipitacion": "Lluvia"},
        },
        "estaciones": [{
            "id_estacion": "x", "tipo_precipitacion": "acumulada",
            "metadata_origen": {
                "nombre": "E", "latitud": -34, "longitud": -58,
            },
            "descarga": {"archivo": "x.xls"},
        }],
    }))
    assert load_catalog(path)[0].tipo_precipitacion == "acumulada"


def test_catalog_requires_explicit_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"datasets": [{"id": "x"}]}))
    with pytest.raises(CatalogError):
        load_catalog(path)
