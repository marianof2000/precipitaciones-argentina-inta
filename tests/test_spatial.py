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
    active_stations,
    coordinate_to_pixel,
    create_spatial_grid,
    cross_validate_idw,
    idw_at_points,
    idw_interpolation,
    interpolate,
    load_territory,
    pixel_center_to_coordinate,
    project_coordinates,
    raster_pixel_for_coordinate,
    validate_active_station_coverage,
)
from precipitaciones_argentina.statistics import (
    generate_log_legend_ticks,
    generate_precipitation_ticks,
    normalize_precipitation_for_color,
    precipitation_global_maximum,
    quarterly_statistics,
    thin_legend_ticks,
)


def test_global_maximum():
    assert precipitation_global_maximum(pd.DataFrame({"precipitacion_mm": [1, 137]})) == 137


def test_ticks_include_exact_maximum():
    assert generate_precipitation_ticks(137, 10) == [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 137,
    ]


def test_color_normalization_linear_and_logarithmic():
    assert normalize_precipitation_for_color(50, 100, "linear") == 0.5
    assert normalize_precipitation_for_color(0, 100, "log") == 0
    assert normalize_precipitation_for_color(100, 100, "log") == 1
    assert normalize_precipitation_for_color(0, 0, "log") == 0
    for scale in ("linear", "log"):
        normalized = [
            normalize_precipitation_for_color(value, 500, scale)
            for value in (10, 50, 100, 500)
        ]
        assert normalized == sorted(normalized) and len(set(normalized)) == 4
    linear_gap = normalize_precipitation_for_color(
        500, 1000, "linear"
    ) - normalize_precipitation_for_color(100, 1000, "linear")
    log_gap = normalize_precipitation_for_color(
        500, 1000, "log"
    ) - normalize_precipitation_for_color(100, 1000, "log")
    assert log_gap < linear_gap
    expected = np.log1p(85.0) / np.log1p(1260.5)
    assert normalize_precipitation_for_color(85.0, 1260.5, "log") == pytest.approx(
        expected
    )


def test_log_legend_ticks_are_real_ordered_values():
    ticks = generate_log_legend_ticks(287)
    assert ticks[0] == 0 and ticks[-1] == 287
    assert ticks == sorted(set(ticks))
    assert all(value <= 287 for value in ticks)


def test_linear_legend_ticks_are_thinned_without_losing_extremes():
    ticks = generate_precipitation_ticks(1260.5, 10)
    thinned = thin_legend_ticks(ticks, 9)
    assert len(thinned) == 9
    assert thinned[0] == 0 and thinned[-1] == 1260.5


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


def test_idw_uses_distance_mask_without_convex_hull_and_preserves_values():
    grid = create_spatial_grid(box(0, 0, 2, 2), 1)
    assert grid.bounds[0] == pytest.approx([-0.5, -0.5], abs=1e-3)
    assert grid.bounds[1] == pytest.approx([2.5, 2.5], abs=1e-3)
    pixel = raster_pixel_for_coordinate(grid, 0.1, 0.1)
    assert (pixel.row, pixel.column) == (2, 0)
    assert (pixel.center_latitude, pixel.center_longitude) == (0.0, 0.0)
    result = idw_interpolation(
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 2.0]),
        np.array([10.0, 20.0, 30.0]),
        grid,
        maximum_distance_km=500,
    )
    assert result.station_count == 3
    assert result.values[0, 0] == 10
    assert np.isfinite(result.values[1, 1])


def test_leaflet_bounds_order_and_coordinate_pixel_roundtrip():
    grid = create_spatial_grid(box(-59, -36, -57, -34), 0.1)
    (south, west), (north, east) = grid.bounds
    assert south < north and west < east
    row, column = coordinate_to_pixel(
        -35.74, -58.05, raster_bounds=grid.bounds,
        shape=(len(grid.latitudes), len(grid.longitudes)),
    )
    latitude, longitude = pixel_center_to_coordinate(
        row, column, raster_bounds=grid.bounds,
        shape=(len(grid.latitudes), len(grid.longitudes)),
    )
    assert abs(latitude - -35.74) <= 0.05 + 1e-12
    assert abs(longitude - -58.05) <= 0.05 + 1e-12


def test_meshgrid_flatten_reshape_keeps_latitude_rows_and_longitude_columns():
    longitudes = np.array([-59.0, -58.0, -57.0])
    latitudes = np.array([-36.0, -35.0, -34.0])
    grid_x, grid_y = np.meshgrid(longitudes, latitudes)
    reconstructed_x = grid_x.ravel(order="C").reshape(grid_x.shape, order="C")
    reconstructed_y = grid_y.ravel(order="C").reshape(grid_y.shape, order="C")
    assert reconstructed_x[0].tolist() == longitudes.tolist()
    assert reconstructed_y[:, 0].tolist() == latitudes.tolist()


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
    stations = np.array([[-59.0, -35.0], [-57.0, -35.0]])
    targets = np.array([[-58.0, -35.0]])
    assert nearest_station_distance_km(stations, targets)[0] == pytest.approx(
        91.3, rel=0.03
    )
    assert stations_within_radius(stations, targets, 120)[0] == 2


def test_active_station_with_valid_rain_is_included_and_covered_at_its_location():
    frame = pd.DataFrame({
        "dataset_id": ["chascomus", "missing"],
        "estacion": ["Chascomus - EEA Cuenca Salado", "Sin dato"],
        "latitud": [-35.58, -35.0], "longitud": [-58.01, -58.0],
        "precipitacion_mm": [184.2, np.nan], "provincia": ["Sin asignar", "P"],
        "fuente": ["F", "F"],
    })
    territory = box(-59, -36, -57, -34)
    active = active_stations(frame, territory=territory)
    assert active["dataset_id"].tolist() == ["chascomus"]
    station = active[["longitud", "latitud"]].to_numpy(float)
    assert nearest_station_distance_km(station, station)[0] == pytest.approx(0)
    validate_active_station_coverage(active, territory, 350)
    grid = create_spatial_grid(box(-58.01, -35.58, -57.01, -34.58), 1)
    result = idw_interpolation(
        active["longitud"].to_numpy(), active["latitud"].to_numpy(),
        active["precipitacion_mm"].to_numpy(), grid,
        minimum_stations=1, maximum_distance_km=350,
    )
    assert result.valid_mask[0, 0]
    assert result.values[0, 0] == pytest.approx(184.2)
    exact = idw_at_points(
        active["longitud"].to_numpy(), active["latitud"].to_numpy(),
        active["precipitacion_mm"].to_numpy(), np.array([-58.01]),
        np.array([-35.58]), territory,
    )
    assert exact.distances_km[0] == pytest.approx(0)
    assert exact.territory_mask[0] and exact.distance_mask[0] and exact.valid_mask[0]
    assert exact.values[0] == pytest.approx(184.2)


def test_coastal_grid_mask_uses_point_center_not_whole_cell():
    narrow_land = box(0, 0, 0.6, 1)
    grid = create_spatial_grid(narrow_land, 0.5)
    longitude_index = int(np.where(grid.longitudes == 0.5)[0][0])
    latitude_index = int(np.abs(grid.latitudes - 0.5).argmin())
    assert grid.territory_mask[latitude_index, longitude_index]


def test_metric_projection_is_not_degree_distance():
    projected = project_coordinates(np.array([[-58.0, -35.0], [-58.0, -34.0]]))
    distance_km = np.linalg.norm(projected[1] - projected[0]) / 1000
    assert distance_km == pytest.approx(111, rel=0.03)


def test_interpolation_territory_excludes_antarctica():
    path = Path(__file__).parents[1] / "assets" / "argentina_provincias.geojson"
    assert load_territory(path).bounds[1] > -56


def test_chascomus_raster_alignment_at_point_one_degree_resolution():
    path = Path(__file__).parents[1] / "assets" / "argentina_provincias.geojson"
    grid = create_spatial_grid(load_territory(path), 0.1)
    pixel = raster_pixel_for_coordinate(grid, -58.05, -35.74)
    assert abs(pixel.center_latitude - -35.74) <= 0.05 + 1e-12
    assert abs(pixel.center_longitude - -58.05) <= 0.05 + 1e-12


def test_idw_leave_one_out_metrics_are_finite():
    result = cross_validate_idw(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([10.0, 20.0, 30.0]),
    )
    assert result.sample_count == 3
    assert np.isfinite(result.mae) and np.isfinite(result.rmse)
