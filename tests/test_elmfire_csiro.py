from __future__ import annotations

import numpy as np
import pytest

from fire_sim.elmfire_csiro import (
    CUSTOM_FUEL,
    FIRES,
    NONBURNABLE_FUEL,
    Conditions,
    custom_fuel_row,
    fuel_map,
    head_positions,
    lighting_points,
    patchwork_fuel_map,
    plot_average_grass,
    wind_2m_maps,
    wind_at_20ft_mph,
)

C064, F19 = FIRES


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


def test_field_wind_has_a_map_every_5_seconds_until_the_end():
    assert len(wind_2m_maps(F19, Conditions.FIELD)) == 37


def test_field_wind_varies_across_the_plot_between_the_upwind_corners():
    first = wind_2m_maps(F19, Conditions.FIELD)[0]
    # At lighting the right (NE) corner read 5.2 m/s and the left (SE) corner 3.9 m/s.
    assert (first[0, 0], first[-1, 0]) == (pytest.approx(3.9), pytest.approx(5.2))


def test_patchwork_uses_all_16_field_samples():
    fuels = patchwork_fuel_map(C064, Conditions.FIELD)
    assert set(np.unique(fuels)) == {*range(CUSTOM_FUEL, CUSTOM_FUEL + 16), NONBURNABLE_FUEL}
