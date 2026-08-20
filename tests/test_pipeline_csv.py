from pathlib import Path

from precipitaciones_argentina import cli, config
from precipitaciones_argentina.catalog import load_csv_manifest


def test_operational_source_contains_no_excel_reader():
    source_dir = Path(__file__).parents[1] / "src" / "precipitaciones_argentina"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_dir.glob("*.py"))
    forbidden = ("read_excel", "ExcelFile", "xlrd", "openpyxl")
    assert all(token not in source for token in forbidden)


def test_pipeline_reads_consolidated_csv_end_to_end(tmp_path, monkeypatch):
    def fake_map(frame, **kwargs):
        output = tmp_path / "index.html"
        output.write_text(f"{len(frame)}", encoding="utf-8")
        return output

    monkeypatch.setattr(cli, "generate_map", fake_map)
    monkeypatch.setattr(cli, "evaluate_idw", lambda frame: {})
    quarterly, summary = cli.run_pipeline()
    assert not quarterly.empty
    assert summary.source_file == "datos/estaciones.csv"
    manifest = load_csv_manifest(config.OBSERVATIONS_MANIFEST)
    assert summary.records_read == manifest["cantidad_registros"]
    assert config.OUTPUT_PARQUET.is_file()
