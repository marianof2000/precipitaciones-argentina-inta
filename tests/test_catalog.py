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


def test_catalog_requires_explicit_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"datasets": [{"id": "x"}]}))
    with pytest.raises(CatalogError):
        load_catalog(path)

