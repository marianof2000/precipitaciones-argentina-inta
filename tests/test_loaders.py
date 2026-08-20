import hashlib

import pandas as pd
import pytest

from precipitaciones_argentina.loaders import DatasetLoadError, read_observations_csv


def csv_and_manifest(tmp_path):
    path = tmp_path / "estaciones.csv"
    pd.DataFrame({"id_estacion": ["A1"], "fecha": ["2024-01-01"],
                  "precipitacion_pluviometrica": [0.0]}).to_csv(path, index=False)
    manifest = {"tamano_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cantidad_registros": 1, "cantidad_columnas": 3,
                "columnas": ["id_estacion", "fecha", "precipitacion_pluviometrica"]}
    return path, manifest


def test_csv_is_validated_against_manifest(tmp_path):
    path, manifest = csv_and_manifest(tmp_path)
    assert len(read_observations_csv(path, manifest)) == 1


def test_sha_mismatch_is_fatal(tmp_path):
    path, manifest = csv_and_manifest(tmp_path)
    manifest["sha256"] = "0" * 64
    with pytest.raises(DatasetLoadError, match="SHA-256"):
        read_observations_csv(path, manifest)


def test_missing_csv_is_fatal(tmp_path):
    with pytest.raises(DatasetLoadError, match="No se encontró"):
        read_observations_csv(tmp_path / "missing.csv", {})
