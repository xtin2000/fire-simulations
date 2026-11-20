from __future__ import annotations

import numpy as np
import pytest

from fire_sim.elmfire_csiro import FIRES, Conditions, Fuel, grid
from fire_sim.wrf_sfire_csiro import (
    IGNITION_START_S,
    REFINEMENT,
    arrival_on_elmfire_grid,
    category_map,
    domain_cells,
    fire_namelist,
    fuel_file,
    ignition_lines,
    input_namelist,
    to_wrf_xy,
)

C064, F19 = FIRES


def test_domain_covers_the_whole_elmfire_map():
    x, y = grid(F19)
    cells_x, cells_y = domain_cells(F19)
    assert (cells_x * 10 >= len(x), cells_y * 10 >= len(y)) == (True, True)


def test_wrf_origin_is_the_south_west_corner_of_the_elmfire_map():
    x, y = grid(C064)
    assert to_wrf_xy(C064, x[0] - 0.5, y[-1] - 0.5) == (0.0, 0.0)


def test_lighting_lines_start_at_the_lighting_centre_and_end_at_its_ends():
    first, second = ignition_lines(C064)
    assert (first["start_y"], first["end_y"], second["end_y"]) == (
        pytest.approx(60.0),
        pytest.approx(85.0),
        pytest.approx(35.0),
    )


def test_lighting_takes_as_long_as_walking_the_half_line():
    first, _ = ignition_lines(C064)
    assert first["end_time"] - IGNITION_START_S == pytest.approx(25 / C064.field_walking_speed_m_s)


def test_namelist_gives_each_ignition_line_its_own_numbers():
    text = input_namelist(C064)
    assert "fire_ignition_start_x2" in text


def test_fuel_file_header_matches_the_fire_map_size():
    cells_x, cells_y = domain_cells(C064)
    header = fuel_file(C064, Fuel.STANDARD, Conditions.FIELD).splitlines()[0]
    assert header == f"{(cells_x + 1) * REFINEMENT} {(cells_y + 1) * REFINEMENT}"


def test_fuel_file_has_one_row_per_west_east_cell():
    cells_x, _ = domain_cells(C064)
    rows = fuel_file(C064, Fuel.STANDARD, Conditions.FIELD).splitlines()[1:]
    assert len(rows) == (cells_x + 1) * REFINEMENT


def test_measured_grass_leaves_bare_ground_as_the_category_after_the_grass():
    categories = category_map(C064, Fuel.MEASURED, Conditions.FIELD)
    assert categories.max() == 2


def test_measured_grass_fuel_table_has_a_bare_ground_entry():
    assert "no_fuel_cat = 2" in fire_namelist(C064, Fuel.MEASURED)


def test_standard_grass_only_sets_the_moisture():
    assert "fuel_categories" not in fire_namelist(C064, Fuel.STANDARD)


def test_arrival_times_come_back_north_first_and_count_from_lighting():
    x, y = grid(C064)
    ignition = np.full((len(y) + 5, len(x) + 5), 1e9)
    level_set = np.ones_like(ignition)
    ignition[0, 0], level_set[0, 0] = IGNITION_START_S + 12.0, -1.0
    arrival = arrival_on_elmfire_grid(C064, ignition, level_set)
    assert arrival[-1, 0] == pytest.approx(12.0)
    assert np.isnan(arrival[0, 0])
