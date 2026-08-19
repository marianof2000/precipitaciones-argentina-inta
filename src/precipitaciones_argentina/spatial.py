"""Geometrías, grillas e interpolación espacial IDW."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import covers, intersects_xy, points
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config


@dataclass(frozen=True)
class SpatialGrid:
    """Grilla regular y máscara territorial en WGS84."""

    longitudes: np.ndarray
    latitudes: np.ndarray
    territory_mask: np.ndarray
    territory: BaseGeometry

    @property
    def bounds(self) -> list[list[float]]:
        """Bordes externos de píxeles Leaflet, no centros de la grilla."""
        longitude_step = float(np.diff(self.longitudes).mean())
        mercator_latitudes = latitude_to_web_mercator_y(self.latitudes)
        mercator_step = float(np.diff(mercator_latitudes).mean())
        return [
            [
                float(
                    web_mercator_y_to_latitude(
                        mercator_latitudes.min() - mercator_step / 2
                    )
                ),
                float(self.longitudes.min() - longitude_step / 2),
            ],
            [
                float(
                    web_mercator_y_to_latitude(
                        mercator_latitudes.max() + mercator_step / 2
                    )
                ),
                float(self.longitudes.max() + longitude_step / 2),
            ],
        ]


@dataclass(frozen=True)
class InterpolationResult:
    """Resultado IDW y trazabilidad de su cobertura."""

    values: np.ndarray
    valid_mask: np.ndarray
    station_count: int
    diagnostics: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossValidationResult:
    """Métricas de validación leave-one-station-out."""

    mae: float
    rmse: float
    sample_count: int


@dataclass(frozen=True)
class PointInterpolationResult:
    """IDW y causas de validez evaluadas en coordenadas puntuales."""

    values: np.ndarray
    territory_mask: np.ndarray
    distances_km: np.ndarray
    distance_mask: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class RasterPixel:
    """Píxel final de Leaflet que contiene una coordenada WGS84."""

    row: int
    column: int
    center_latitude: float
    center_longitude: float
    west: float
    east: float
    south: float
    north: float


def latitude_to_web_mercator_y(latitude: np.ndarray | float) -> np.ndarray:
    """Convierte latitud WGS84 a la ordenada normalizada de Web Mercator."""
    latitude_array = np.asarray(latitude, dtype=float)
    clipped = np.clip(latitude_array, -85.05112878, 85.05112878)
    radians = np.deg2rad(clipped)
    return np.log(np.tan(np.pi / 4 + radians / 2))


def web_mercator_y_to_latitude(y: np.ndarray | float) -> np.ndarray:
    """Convierte la ordenada normalizada de Web Mercator a latitud WGS84."""
    return np.rad2deg(2 * np.arctan(np.exp(np.asarray(y, dtype=float))) - np.pi / 2)


def project_coordinates(
    coordinates: np.ndarray,
    source_crs: str = "EPSG:4326",
    target_crs: str = config.DISTANCE_CRS,
) -> np.ndarray:
    """Proyecta pares longitud/latitud a un CRS métrico apto para Argentina."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size == 0:
        return np.empty((0, 2), dtype=float)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    return np.column_stack((x, y))


def active_stations(
    quarter_data: pd.DataFrame,
    value_column: str = "precipitacion_mm",
    territory: BaseGeometry | None = None,
) -> pd.DataFrame:
    """Selecciona una sola fuente de estaciones activas para una variable y período."""
    coordinates = quarter_data[["longitud", "latitud"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        pd.to_numeric(quarter_data[value_column], errors="coerce").notna()
        & coordinates["longitud"].between(-180, 180)
        & coordinates["latitud"].between(-90, 90)
    )
    if territory is not None:
        valid &= intersects_xy(
            territory,
            coordinates["longitud"].to_numpy(),
            coordinates["latitud"].to_numpy(),
        )
    return quarter_data.loc[valid].drop_duplicates("dataset_id").copy()


def validate_active_station_coverage(
    stations: pd.DataFrame,
    territory: BaseGeometry,
    maximum_distance_km: float,
) -> None:
    """Garantiza que cada estación activa pertenezca a su cobertura potencial."""
    if stations.empty:
        return
    coordinates = stations[["longitud", "latitud"]].to_numpy(float)
    if not intersects_xy(territory, coordinates[:, 0], coordinates[:, 1]).all():
        raise ValueError("Una estación activa quedó fuera de la máscara territorial")
    metric = project_coordinates(coordinates)
    own_distances, _ = cKDTree(metric).query(metric, k=1)
    if not np.allclose(own_distances, 0, atol=1e-6):
        raise ValueError("Una estación activa no tiene distancia propia igual a cero")
    if maximum_distance_km <= 0:
        raise ValueError("La distancia máxima de interpolación debe ser positiva")


def load_territory(path: Path, target_crs: str = "EPSG:4326") -> BaseGeometry:
    """Carga el territorio continental e insular próximo apto para interpolación.

    El GeoJSON incluye Antártida e islas del Atlántico Sur dentro de Tierra del Fuego.
    Sin estaciones válidas allí, esos componentes no deben ampliar la máscara IDW.
    """
    provinces = gpd.read_file(path)
    if provinces.crs is None:
        provinces = provinces.set_crs(target_crs)
    else:
        provinces = provinces.to_crs(target_crs)
    geometries: list[BaseGeometry] = []
    for _, province in provinces.iterrows():
        geometry = province.geometry
        if str(province.get("nombre", "")).startswith("Tierra del Fuego"):
            components = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
            geometry = unary_union(
                [
                    component
                    for component in components
                    if component.area > 0.05
                    and component.bounds[3] > -56
                    and component.centroid.x < -63
                ]
            )
        geometries.append(geometry)
    return unary_union(geometries)


def create_spatial_grid(territory: BaseGeometry, resolution: float) -> SpatialGrid:
    """Crea centros regulares en Web Mercator, la proyección usada por Leaflet."""
    if resolution <= 0:
        raise ValueError("La resolución debe ser positiva")
    minimum_x, minimum_y, maximum_x, maximum_y = territory.bounds
    longitudes = np.arange(minimum_x, maximum_x + resolution, resolution)
    minimum_mercator_y = float(latitude_to_web_mercator_y(minimum_y))
    maximum_mercator_y = float(latitude_to_web_mercator_y(maximum_y))
    # La separación angular máxima se fija en el extremo norte, donde un mismo
    # paso Mercator abarca más grados dentro del dominio argentino.
    northern_step = float(
        latitude_to_web_mercator_y(maximum_y + resolution)
        - latitude_to_web_mercator_y(maximum_y)
    )
    latitude_count = (
        int(np.ceil((maximum_mercator_y - minimum_mercator_y) / northern_step)) + 1
    )
    mercator_y = np.linspace(minimum_mercator_y, maximum_mercator_y, latitude_count)
    latitudes = web_mercator_y_to_latitude(mercator_y)
    grid_x, grid_y = np.meshgrid(longitudes, latitudes)
    mask = covers(territory, points(grid_x, grid_y))
    return SpatialGrid(longitudes, latitudes, mask, territory)


def raster_pixel_for_coordinate(
    grid: SpatialGrid, longitude: float, latitude: float
) -> RasterPixel:
    """Localiza el píxel que Leaflet muestra para una coordenada."""
    shape = (len(grid.latitudes), len(grid.longitudes))
    row, column = coordinate_to_pixel(
        latitude, longitude, raster_bounds=grid.bounds, shape=shape
    )
    center_latitude, center_longitude = pixel_center_to_coordinate(
        row, column, raster_bounds=grid.bounds, shape=shape
    )
    longitude_step = float(np.diff(grid.longitudes).mean())
    (raster_south, _), (raster_north, _) = grid.bounds
    south_y = float(latitude_to_web_mercator_y(raster_south))
    north_y = float(latitude_to_web_mercator_y(raster_north))
    row_step = (north_y - south_y) / shape[0]
    pixel_south = float(web_mercator_y_to_latitude(north_y - (row + 1) * row_step))
    pixel_north = float(web_mercator_y_to_latitude(north_y - row * row_step))
    return RasterPixel(
        row, column, center_latitude, center_longitude,
        center_longitude - longitude_step / 2,
        center_longitude + longitude_step / 2,
        pixel_south,
        pixel_north,
    )


def coordinate_to_pixel(
    latitude: float,
    longitude: float,
    *,
    raster_bounds: list[list[float]],
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Convierte WGS84 al píxel Leaflet: fila norte→sur, columna oeste→este."""
    (south, west), (north, east) = raster_bounds
    rows, columns = shape
    if not (south < north and west < east and rows > 0 and columns > 0):
        raise ValueError("Bounds u organización del ráster inválidos")
    if not (west <= longitude <= east and south <= latitude <= north):
        raise ValueError("La coordenada está fuera de los bounds del ráster")
    north_y = float(latitude_to_web_mercator_y(north))
    south_y = float(latitude_to_web_mercator_y(south))
    latitude_y = float(latitude_to_web_mercator_y(latitude))
    row = min(rows - 1, int((north_y - latitude_y) / (north_y - south_y) * rows))
    column = min(columns - 1, int((longitude - west) / (east - west) * columns))
    return row, column


def pixel_center_to_coordinate(
    row: int,
    column: int,
    *,
    raster_bounds: list[list[float]],
    shape: tuple[int, int],
) -> tuple[float, float]:
    """Convierte un píxel Leaflet a la coordenada WGS84 de su centro."""
    (south, west), (north, east) = raster_bounds
    rows, columns = shape
    if not (0 <= row < rows and 0 <= column < columns):
        raise ValueError("Índice de píxel fuera del ráster")
    north_y = float(latitude_to_web_mercator_y(north))
    south_y = float(latitude_to_web_mercator_y(south))
    latitude_y = north_y - (row + 0.5) * (north_y - south_y) / rows
    latitude = web_mercator_y_to_latitude(latitude_y)
    longitude = west + (column + 0.5) * (east - west) / columns
    return float(latitude), float(longitude)


def idw_at_points(
    station_longitudes: np.ndarray,
    station_latitudes: np.ndarray,
    values: np.ndarray,
    target_longitudes: np.ndarray,
    target_latitudes: np.ndarray,
    territory: BaseGeometry,
    *,
    power: float = 2.0,
    maximum_distance_km: float = 350.0,
) -> PointInterpolationResult:
    """Evalúa máscara e IDW en puntos exactos usando distancias métricas."""
    finite = np.isfinite(station_longitudes) & np.isfinite(station_latitudes) & np.isfinite(values)
    stations_wgs84 = np.column_stack((station_longitudes[finite], station_latitudes[finite]))
    observations = values[finite].astype(float)
    unique_stations, inverse = np.unique(stations_wgs84, axis=0, return_inverse=True)
    observations = np.bincount(inverse, weights=observations) / np.bincount(inverse)
    targets_wgs84 = np.column_stack((target_longitudes, target_latitudes))
    territory_mask = covers(territory, points(target_longitudes, target_latitudes))
    result = np.full(len(targets_wgs84), np.nan, dtype=float)
    if not len(unique_stations):
        infinite = np.full(len(targets_wgs84), np.inf)
        false_mask = np.zeros(len(targets_wgs84), dtype=bool)
        return PointInterpolationResult(result, territory_mask, infinite, false_mask, false_mask)
    stations_metric = project_coordinates(unique_stations)
    targets_metric = project_coordinates(targets_wgs84)
    tree = cKDTree(stations_metric)
    nearest_metres, _ = tree.query(targets_metric, k=1)
    distances_km = nearest_metres / 1000
    distance_mask = distances_km <= maximum_distance_km
    valid_mask = territory_mask & distance_mask
    valid_targets = targets_metric[valid_mask]
    if len(valid_targets):
        distances = np.linalg.norm(
            valid_targets[:, None, :] - stations_metric[None, :, :], axis=2
        )
        exact = distances <= 1e-6
        weights = np.divide(
            1.0, np.power(distances, power),
            out=np.zeros_like(distances), where=~exact,
        )
        weight_sums = weights.sum(axis=1)
        estimates = np.divide(
            weights @ observations, weight_sums,
            out=np.zeros(len(valid_targets)), where=weight_sums > 0,
        )
        exact_rows = exact.any(axis=1)
        estimates[exact_rows] = observations[np.argmax(exact[exact_rows], axis=1)]
        result[valid_mask] = estimates
    return PointInterpolationResult(
        result, territory_mask, distances_km, distance_mask, valid_mask
    )


def idw_interpolation(
    station_longitudes: np.ndarray,
    station_latitudes: np.ndarray,
    values: np.ndarray,
    grid: SpatialGrid,
    *,
    power: float = 2.0,
    maximum_distance_km: float = 350.0,
    minimum_stations: int = 3,
) -> InterpolationResult:
    """Interpola dentro del territorio y del umbral métrico a estaciones activas."""
    if power <= 0 or maximum_distance_km <= 0:
        raise ValueError("power y maximum_distance_km deben ser positivos")
    finite = np.isfinite(station_longitudes) & np.isfinite(station_latitudes) & np.isfinite(values)
    points_wgs84 = np.column_stack((station_longitudes[finite], station_latitudes[finite]))
    observations = values[finite].astype(float)
    unique_points_wgs84, inverse = np.unique(points_wgs84, axis=0, return_inverse=True)
    value_sums = np.bincount(inverse, weights=observations)
    observations = value_sums / np.bincount(inverse)
    shape = (len(grid.latitudes), len(grid.longitudes))
    empty = np.full(shape, np.nan, dtype=float)
    if len(unique_points_wgs84) < minimum_stations:
        return InterpolationResult(empty, np.zeros(shape, dtype=bool), len(unique_points_wgs84))

    grid_x, grid_y = np.meshgrid(grid.longitudes, grid.latitudes)
    targets_wgs84 = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    unique_points = project_coordinates(unique_points_wgs84)
    targets = project_coordinates(targets_wgs84)
    tree = cKDTree(unique_points)
    nearest_metres, _ = tree.query(targets, k=1)
    territory_mask = grid.territory_mask.ravel()
    distance_mask = nearest_metres <= maximum_distance_km * 1000
    spatial_mask = territory_mask & distance_mask
    valid_targets = targets[spatial_mask]
    if not len(valid_targets):
        return InterpolationResult(empty, np.zeros(shape, dtype=bool), len(unique_points))
    delta_x = valid_targets[:, None, 0] - unique_points[None, :, 0]
    delta_y = valid_targets[:, None, 1] - unique_points[None, :, 1]
    distances = np.hypot(delta_x, delta_y)
    exact = distances <= 1e-12
    weights = np.divide(
        1.0,
        np.power(distances, power),
        out=np.zeros_like(distances),
        where=~exact,
    )
    weight_sums = weights.sum(axis=1)
    interpolated = np.divide(
        weights @ observations,
        weight_sums,
        out=np.zeros(len(valid_targets), dtype=float),
        where=weight_sums > 0,
    )
    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        interpolated[exact_rows] = observations[np.argmax(exact[exact_rows], axis=1)]
    flat = empty.ravel()
    flat[spatial_mask] = interpolated
    valid_mask = np.zeros(flat.shape, dtype=bool)
    valid_mask[spatial_mask] = True
    diagnostics = {
        "total_puntos": int(len(targets)),
        "dentro_territorio": int(territory_mask.sum()),
        "dentro_distancia": int(distance_mask.sum()),
        "validos_idw": int(spatial_mask.sum()),
        "nan_territorio": int((~territory_mask).sum()),
        "nan_distancia": int((territory_mask & ~distance_mask).sum()),
        "nan_inesperados": int(np.isnan(flat[spatial_mask]).sum()),
    }
    return InterpolationResult(
        flat.reshape(shape), valid_mask.reshape(shape), len(unique_points), diagnostics
    )


def interpolate(
    method: str,
    station_longitudes: np.ndarray,
    station_latitudes: np.ndarray,
    values: np.ndarray,
    grid: SpatialGrid,
    **parameters: float | int,
) -> InterpolationResult:
    """Despacha el método espacial sin ocultar algoritmos aún no implementados."""
    if method.casefold() == "idw":
        return idw_interpolation(
            station_longitudes, station_latitudes, values, grid, **parameters
        )
    raise NotImplementedError(
        f"Método de interpolación no implementado: {method}. "
        "Los candidatos previstos son RBF y Kriging."
    )


def cross_validate_idw(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    values: np.ndarray,
    power: float = 2.0,
) -> CrossValidationResult:
    """Evalúa IDW retirando sucesivamente cada observación válida."""
    finite = np.isfinite(longitudes) & np.isfinite(latitudes) & np.isfinite(values)
    points = project_coordinates(np.column_stack((longitudes[finite], latitudes[finite])))
    observations = values[finite].astype(float)
    estimates: list[float] = []
    actual: list[float] = []
    for index, point in enumerate(points):
        remaining = np.arange(len(points)) != index
        if remaining.sum() < 2:
            continue
        distances = np.hypot(
            points[remaining, 0] - point[0], points[remaining, 1] - point[1]
        )
        remaining_values = observations[remaining]
        exact = distances <= 1e-12
        if exact.any():
            estimate = float(remaining_values[exact].mean())
        else:
            weights = 1 / np.power(distances, power)
            estimate = float(np.average(remaining_values, weights=weights))
        estimates.append(estimate)
        actual.append(float(observations[index]))
    if not estimates:
        return CrossValidationResult(float("nan"), float("nan"), 0)
    errors = np.asarray(estimates) - np.asarray(actual)
    return CrossValidationResult(
        mae=float(np.abs(errors).mean()),
        rmse=float(np.sqrt(np.square(errors).mean())),
        sample_count=len(errors),
    )
