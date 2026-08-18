"""Geometrías, grillas e interpolación espacial IDW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

EARTH_KM_PER_DEGREE = 111.32


@dataclass(frozen=True)
class SpatialGrid:
    """Grilla regular y máscara territorial en WGS84."""

    longitudes: np.ndarray
    latitudes: np.ndarray
    territory_mask: np.ndarray
    territory: BaseGeometry

    @property
    def bounds(self) -> list[list[float]]:
        """Límites Leaflet en orden sudoeste/noreste."""
        return [
            [float(self.latitudes.min()), float(self.longitudes.min())],
            [float(self.latitudes.max()), float(self.longitudes.max())],
        ]


@dataclass(frozen=True)
class InterpolationResult:
    """Resultado IDW y trazabilidad de su cobertura."""

    values: np.ndarray
    valid_mask: np.ndarray
    station_count: int


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
    """Crea centros de celda regulares limitados al bbox del territorio."""
    if resolution <= 0:
        raise ValueError("La resolución debe ser positiva")
    minimum_x, minimum_y, maximum_x, maximum_y = territory.bounds
    longitudes = np.arange(minimum_x, maximum_x + resolution, resolution)
    latitudes = np.arange(minimum_y, maximum_y + resolution, resolution)
    grid_x, grid_y = np.meshgrid(longitudes, latitudes)
    mask = intersects_xy(territory, grid_x, grid_y)
    return SpatialGrid(longitudes, latitudes, mask, territory)


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
    """Interpola dentro de territorio, convex hull y distancia máxima a estaciones."""
    if power <= 0 or maximum_distance_km <= 0:
        raise ValueError("power y maximum_distance_km deben ser positivos")
    finite = np.isfinite(station_longitudes) & np.isfinite(station_latitudes) & np.isfinite(values)
    points = np.column_stack((station_longitudes[finite], station_latitudes[finite]))
    observations = values[finite].astype(float)
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    value_sums = np.bincount(inverse, weights=observations)
    observations = value_sums / np.bincount(inverse)
    shape = (len(grid.latitudes), len(grid.longitudes))
    empty = np.full(shape, np.nan, dtype=float)
    if len(unique_points) < minimum_stations:
        return InterpolationResult(empty, np.zeros(shape, dtype=bool), len(unique_points))
    hull = MultiPoint(unique_points).convex_hull
    if hull.geom_type != "Polygon" or hull.area == 0:
        return InterpolationResult(empty, np.zeros(shape, dtype=bool), len(unique_points))

    grid_x, grid_y = np.meshgrid(grid.longitudes, grid.latitudes)
    targets = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    tree = cKDTree(unique_points)
    nearest_degrees, _ = tree.query(targets, k=1)
    spatial_mask = (
        grid.territory_mask.ravel()
        & intersects_xy(hull, targets[:, 0], targets[:, 1])
        & (nearest_degrees * EARTH_KM_PER_DEGREE <= maximum_distance_km)
    )
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
    interpolated = (weights @ observations) / weights.sum(axis=1)
    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        interpolated[exact_rows] = observations[np.argmax(exact[exact_rows], axis=1)]
    flat = empty.ravel()
    flat[spatial_mask] = interpolated
    valid_mask = np.zeros(flat.shape, dtype=bool)
    valid_mask[spatial_mask] = True
    return InterpolationResult(flat.reshape(shape), valid_mask.reshape(shape), len(unique_points))
