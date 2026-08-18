from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from precipitaciones_argentina.coverage import (
    nearest_station_distance_km,
    stations_within_radius,
)
from precipitaciones_argentina.spatial import (
    create_spatial_grid,
    cross_validate_idw,
    idw_interpolation,
    interpolate,
    load_territory,
)
from precipitaciones_argentina.statistics import (
    generate_precipitation_ticks,
    precipitation_global_maximum,
    quarterly_statistics,
)


def test_global_maximum():
    assert precipitation_global_maximum(pd.DataFrame({"precipitacion_mm": [1, 137]})) == 137


def test_ticks_include_exact_maximum():
    assert generate_precipitation_ticks(137, 10) == [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 137,
    ]


def test_quarterly_statistics():
    frame = pd.DataFrame({
        "periodo": ["2024-T1", "2024-T1"],
        "cantidad_observaciones": [10, 20],
        "estacion": ["A", "B"],
        "dataset_id": ["a", "b"],
        "fuente": ["F", "F"],
        "precipitacion_mm": [10.0, 30.0],
    })
    result = quarterly_statistics(frame).iloc[0]
    assert result["cantidad_observaciones"] == 30
    assert result["cantidad_estaciones"] == 2
    assert result["precipitacion_media"] == 20


def test_idw_is_masked_by_hull_and_preserves_station_values():
    grid = create_spatial_grid(box(0, 0, 2, 2), 1)
    result = idw_interpolation(
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 2.0]),
        np.array([10.0, 20.0, 30.0]),
        grid,
        maximum_distance_km=500,
    )
    assert result.station_count == 3
    assert result.values[0, 0] == 10
    assert np.isnan(result.values[2, 2])


def test_idw_requires_enough_non_collinear_stations():
    grid = create_spatial_grid(box(0, 0, 2, 2), 1)
    result = idw_interpolation(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([10.0, 20.0]), grid
    )
    assert not result.valid_mask.any()


def test_interpolation_dispatcher_rejects_unimplemented_methods():
    grid = create_spatial_grid(box(0, 0, 2, 2), 1)
    with pytest.raises(NotImplementedError, match="Kriging"):
        interpolate(
            "kriging", np.array([0.0]), np.array([0.0]), np.array([1.0]), grid
        )


def test_coverage_metrics():
    stations = np.array([[0.0, 0.0], [2.0, 0.0]])
    targets = np.array([[1.0, 0.0]])
    assert nearest_station_distance_km(stations, targets)[0] == 111.32
    assert stations_within_radius(stations, targets, 120)[0] == 2


def test_interpolation_territory_excludes_antarctica():
    path = Path(__file__).parents[1] / "assets" / "argentina_provincias.geojson"
    assert load_territory(path).bounds[1] > -56


def test_idw_leave_one_out_metrics_are_finite():
    result = cross_validate_idw(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([10.0, 20.0, 30.0]),
    )
    assert result.sample_count == 3
    assert np.isfinite(result.mae) and np.isfinite(result.rmse)
