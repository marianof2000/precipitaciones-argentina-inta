"""Indicadores de cobertura independientes de la precipitación."""

import numpy as np
from scipy.spatial import cKDTree

from .spatial import EARTH_KM_PER_DEGREE


def nearest_station_distance_km(
    station_coordinates: np.ndarray, target_coordinates: np.ndarray
) -> np.ndarray:
    """Calcula distancia aproximada a la estación más cercana en WGS84."""
    if len(station_coordinates) == 0:
        return np.full(len(target_coordinates), np.inf)
    distances, _ = cKDTree(station_coordinates).query(target_coordinates, k=1)
    return distances * EARTH_KM_PER_DEGREE


def stations_within_radius(
    station_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """Cuenta estaciones dentro de un radio aproximado para cada destino."""
    if radius_km <= 0:
        raise ValueError("El radio debe ser positivo")
    if len(station_coordinates) == 0:
        return np.zeros(len(target_coordinates), dtype=int)
    neighbours = cKDTree(station_coordinates).query_ball_point(
        target_coordinates, radius_km / EARTH_KM_PER_DEGREE
    )
    return np.fromiter((len(items) for items in neighbours), dtype=int)
