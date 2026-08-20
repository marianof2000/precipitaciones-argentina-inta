"""Configuración central del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datos"
STATION_CATALOG_FILE = DATA_DIR / "estaciones.json"
OBSERVATIONS_CSV = DATA_DIR / "estaciones.csv"
OBSERVATIONS_MANIFEST = DATA_DIR / "estaciones_csv.json"
VERIFY_CSV_SHA256 = True
ASSETS_DIR = PROJECT_ROOT / "assets"
PROVINCES_GEOJSON = ASSETS_DIR / "argentina_provincias.geojson"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_HTML = OUTPUT_DIR / "index.html"
OUTPUT_PARQUET = OUTPUT_DIR / "datos_normalizados.parquet"
OUTPUT_AUDIT = OUTPUT_DIR / "auditoria.json"
OUTPUT_IDW_EVALUATION = OUTPUT_DIR / "evaluacion_idw.json"

TARGET_CRS = "EPSG:4326"
DISTANCE_CRS = "+proj=aeqd +lat_0=-34 +lon_0=-63 +datum=WGS84 +units=m +no_defs"
ARGENTINA_CENTER = (-38.4161, -63.6167)
DEFAULT_ZOOM = 4
PRECIPITATION_MIN = 0.0
PRECIPITATION_STEP = 10.0
INTERPOLATION_METHOD = "idw"
GRID_RESOLUTION = 0.1  # grados; configurable según costo y detalle requerido
SPATIAL_DEBUG = False
SPATIAL_DEBUG_PERIOD = "2015-T3"
IDW_POWER = 2.0
IDW_DEFAULT_OPACITY = 0.45
if not 0 <= IDW_DEFAULT_OPACITY <= 1:
    raise ValueError("IDW_DEFAULT_OPACITY debe estar entre 0 y 1")
MIN_INTERPOLATION_STATIONS = 3
# Umbral provisional: debe calibrarse según densidad y separación de la red.
MAX_INTERPOLATION_DISTANCE_KM = 350.0
COVERAGE_RADIUS_KM = 250.0
GEOJSON_SIMPLIFICATION_TOLERANCE = 0.01
AGGREGATION_METHOD = "sum"
if AGGREGATION_METHOD != "sum":
    raise ValueError("La precipitación acumulada trimestral requiere AGGREGATION_METHOD='sum'")

# Normal climatológica configurable. 1991–2020 sigue el período estándar de 30 años de la OMM.
CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020
MIN_HISTORICAL_YEARS = 5
RELATIVE_ANOMALY_COLOR_LIMIT = 200.0
