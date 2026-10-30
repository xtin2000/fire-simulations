from __future__ import annotations

import numpy as np
import pytest

from fire_sim.burn_scenarios import (
    atv_crew_size,
    block_mask,
    build_input,
    fuel_depth_field,
    ignition_times,
    sample_burn,
)


@pytest.fixture(params=range(6))
def spec(request):
    return sample_burn(request.param, base_seed=7000)


def test_sampling_is_reproducible_from_seed():
    assert sample_burn(11, 7000) == sample_burn(11, 7000)


def test_fuel_and_ignition_stay_inside_the_burn_block(spec):
    outside = ~block_mask(spec)
    assert np.all(fuel_depth_field(spec)[outside] == 0)
    assert not np.isfinite(ignition_times(spec)[outside]).any()


def test_hand_ignition_only_on_burnable_cells(spec):
    lit = np.isfinite(ignition_times(spec))
    assert np.all(fuel_depth_field(spec)[lit] > 0)


def test_crew_size_finishes_lighting_within_target_when_uncapped():
    strips, pass_s, minutes = 30, 400.0, 40.0
    atvs = atv_crew_size(strips, pass_s, minutes)
    assert np.ceil(strips / atvs) * pass_s <= minutes * 60 + pass_s


def test_every_burnable_cell_gets_exactly_one_vent(spec):
    text = build_input(spec)
    vents = [line for line in text.splitlines() if line.startswith("&VENT XB=")]
    covered = 0
    for line in vents:
        x0, x1 = (float(v) for v in line.split("XB=")[1].split(",")[:2])
        covered += round((x1 - x0) / 8.0)
    assert covered == int((fuel_depth_field(spec) > 0).sum())


def test_every_scenario_lights_at_least_one_cell():
    """A point that lands on bare ground must still light the nearest burnable cell."""
    for index in range(300):
        spec = sample_burn(index, base_seed=7000)
        assert np.isfinite(ignition_times(spec)).any(), spec.scenario_id
