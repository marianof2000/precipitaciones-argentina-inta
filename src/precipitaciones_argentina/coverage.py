"""Indicadores de cobertura independientes de la precipitación."""

import numpy as np
from scipy.spatial import cKDTree

from .spatial import project_coordinates


def nearest_station_distance_km(
    station_coordinates: np.ndarray, target_coordinates: np.ndarray
) -> np.ndarray:
    """Calcula distancia métrica a la estación más cercana."""
    if len(station_coordinates) == 0:
        return np.full(len(target_coordinates), np.inf)
    stations_metric = project_coordinates(station_coordinates)
    targets_metric = project_coordinates(target_coordinates)
    distances, _ = cKDTree(stations_metric).query(targets_metric, k=1)
    return distances / 1000


def stations_within_radius(
    station_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """Cuenta estaciones dentro de un radio métrico para cada destino."""
    if radius_km <= 0:
        raise ValueError("El radio debe ser positivo")
    if len(station_coordinates) == 0:
        return np.zeros(len(target_coordinates), dtype=int)
    stations_metric = project_coordinates(station_coordinates)
    targets_metric = project_coordinates(target_coordinates)
    neighbours = cKDTree(stations_metric).query_ball_point(
        targets_metric, radius_km * 1000
    )
    return np.fromiter((len(items) for items in neighbours), dtype=int)
