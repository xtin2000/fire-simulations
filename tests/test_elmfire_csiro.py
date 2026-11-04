from __future__ import annotations

import numpy as np
import pytest

from fire_sim.annaburroo import ANNABURROO_DIR
from fire_sim.elmfire_csiro import (
    CUSTOM_FUEL,
    FIRES,
    NONBURNABLE_FUEL,
    Conditions,
    _grid,
    added_fire,
    all_fires,
    custom_fuel_row,
    fuel_map,
    head_distances,
    head_positions,
    lighting_points,
    patchwork_fuel_map,
    plot_average_grass,
    station_directions,
    wind_2m_maps,
    wind_at_20ft_mph,
)

C064, F19 = FIRES
needs_csiro_data = pytest.mark.skipif(
    not (ANNABURROO_DIR / "Burn_Period_Data.csv").exists(),
    reason="run `python -m fire_sim.annaburroo download` first",
)


def test_wind_at_20ft_is_faster_than_at_2m():
    assert wind_at_20ft_mph(4.6) == pytest.approx(4.6 * 1.2655 / 0.44704, rel=1e-3)


def test_custom_fuel_row_converts_measured_grass_to_us_units():
    fields = custom_fuel_row(CUSTOM_FUEL, plot_average_grass(C064)).split(",")
    assert (fields[0], fields[3], fields[8], fields[11]) == (
        str(CUSTOM_FUEL),
        "0.05796",
        "2978",
        "0.689",
    )


def test_lighting_starts_in_the_centre_and_spreads_at_one_metre_per_second():
    times = {y: t for _, y, t in lighting_points(C064)}
    assert (times[0.0], times[5.0], times[-25.0]) == (0.0, 0.0, 20.0)


def test_fuel_map_has_grass_on_plot_and_bare_ground_around_it():
    fuels = fuel_map(C064, fuel_model=1)
    assert fuels.shape == (120, 120)
    assert set(np.unique(fuels)) == {1, NONBURNABLE_FUEL}


def test_head_position_is_furthest_burned_cell_in_the_head_strip():
    arrival = np.full((120, 120), np.nan)
    arrival[62, 10:41] = np.arange(31)  # row 62 has y = -2.5, column 10 starts at x = 0
    assert head_positions(C064, arrival, np.array([30.0]))[0] == pytest.approx(31.0)


def test_long_lighting_lines_stay_within_elmfires_ignition_limit():
    assert all(len(lighting_points(fire)) <= 100 for fire in FIRES)


def test_field_lighting_starts_at_the_centre_and_walks_at_the_measured_pace():
    times = {y: t for _, y, t in lighting_points(F19, Conditions.FIELD)}
    assert (times[0.0], times[86.0]) == (0.0, pytest.approx(86 * 56 / 88))


@needs_csiro_data
def test_field_wind_has_a_map_every_5_seconds_until_the_end():
    assert len(wind_2m_maps(F19, Conditions.FIELD)) == 37


@needs_csiro_data
def test_field_wind_varies_across_the_plot_between_the_upwind_corners():
    first = wind_2m_maps(F19, Conditions.FIELD)[0]
    # At lighting the right (NE) corner read 5.2 m/s and the left (SE) corner 3.9 m/s.
    assert (first[0, 0], first[-1, 0]) == (pytest.approx(3.9), pytest.approx(5.2))


def test_patchwork_uses_all_16_field_samples():
    fuels = patchwork_fuel_map(C064, Conditions.FIELD)
    assert set(np.unique(fuels)) == {*range(CUSTOM_FUEL, CUSTOM_FUEL + 16), NONBURNABLE_FUEL}


@needs_csiro_data
def test_off_centre_tilted_lighting_line_keeps_its_centre_and_tilt():
    c112 = added_fire("C112")
    points = lighting_points(c112, Conditions.FIELD_STATION)
    centre = next(p for p in points if p[2] == 0.0)
    assert centre[:2] == (c112.lighting_x_m, c112.lighting_centre_y_m)
    north = min(points, key=lambda p: p[1])
    assert north[0] > centre[0]


@needs_csiro_data
def test_every_ignition_point_lies_on_burnable_ground():
    for fire in all_fires():
        fuels = fuel_map(fire, 1, Conditions.FIELD_STATION)
        x, y = _grid(fire)
        for px, py, _ in lighting_points(fire, Conditions.FIELD_STATION):
            # A point on a cell edge may land in either neighbour.
            touching = (np.abs(y - py) <= 0.5)[:, None] & (np.abs(x - px) <= 0.5)[None, :]
            assert (fuels[touching] == 1).any(), (fire.name, px, py)


@needs_csiro_data
def test_station_wind_from_the_east_comes_from_the_rasters_west():
    # On F04's day the station wind came from about 80 degrees, behind the east lighting line.
    directions = station_directions(added_fire("F04"))
    assert all(240 < d < 300 for d in directions)


def test_head_distance_follows_a_real_westward_bearing_down_the_raster_x_axis():
    arrival = np.full((120, 120), np.nan)
    arrival[60, 10:41] = np.arange(31)  # column 10 starts at x = 0
    distance = head_distances(C064, arrival, np.array([30.0]), bearing_deg=270.0)[0]
    assert distance == pytest.approx(30.5 - C064.lighting_x_m)
