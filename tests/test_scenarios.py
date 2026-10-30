from __future__ import annotations

import re

import numpy as np
import pytest

from fire_sim.scenarios import (
    GRID_CELLS,
    ScenarioSpec,
    build_input,
    fuel_depth_field,
    sample_scenario,
    wind_ramps,
)


@pytest.fixture
def spec() -> ScenarioSpec:
    return sample_scenario(index=7, base_seed=2026)


def test_sampling_is_reproducible_from_seed():
    assert sample_scenario(3, 2026) == sample_scenario(3, 2026)


def test_sampled_inputs_stay_within_rxcadre_small_plot_ranges():
    specs = [sample_scenario(i, 2026) for i in range(50)]
    assert all(0.24 <= s.fuel_load_kg_m2 <= 0.41 for s in specs)
    assert all(0.064 <= s.dead_moisture <= 0.100 for s in specs)


def test_fuel_depth_field_averages_to_the_scenario_depth(spec):
    spec.has_firebreak = False
    depth = fuel_depth_field(spec)
    assert depth.shape == (GRID_CELLS, GRID_CELLS)
    assert depth.mean() == pytest.approx(spec.mean_fuel_depth_m, rel=0.02)


def test_firebreak_cells_have_no_fuel(spec):
    spec.has_firebreak = True
    assert (fuel_depth_field(spec) == 0).sum() > GRID_CELLS


def test_wind_direction_completes_its_shift_after_ten_minutes(spec):
    times, _, direction = wind_ramps(spec)
    settled = times >= spec.wind_shift_start_s + 600
    expected = (spec.wind_from_deg + spec.wind_shift_deg) % 360
    assert direction[settled] == pytest.approx(expected)


def test_ignition_cells_never_get_a_second_fuel_vent(spec):
    """FDS rejects a whole VENT that overlaps another, so cells must not be listed twice."""
    vents = re.findall(r"&VENT XB=([\d.]+),([\d.]+),([\d.]+),([\d.]+),0,0", build_input(spec))
    covered = np.zeros((GRID_CELLS, GRID_CELLS), dtype=int)
    for x0, x1, y0, y1 in (map(float, v) for v in vents):
        covered[int(y0 // 2) : int(y1 // 2), int(x0 // 2) : int(x1 // 2)] += 1
    assert covered.max() == 1
