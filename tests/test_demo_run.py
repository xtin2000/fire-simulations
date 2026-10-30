from __future__ import annotations

import re

import pytest

from fire_sim.demo_run import build_input, predicted_speeds, random_inputs


def test_s5_grass_head_speed_matches_the_hand_calculation():
    """S5 grass: no-wind 0.02423 m/s (FDS log) gives a 0.127 m/s head fire at 2.8 m/s wind."""
    assert predicted_speeds(0.02423)["head_m_s"] == pytest.approx(0.1266, abs=5e-4)


def test_random_inputs_are_reproducible_from_the_seed():
    assert random_inputs(1199) == random_inputs(1199)


def test_grass_patches_cover_every_square_except_the_lit_one_exactly_once():
    text = build_input(random_inputs(1199), "demo")
    covered = 0.0
    for x0, x1, y0, y1 in re.findall(r"&VENT XB=([\d.]+),([\d.]+),([\d.]+),([\d.]+),0,0", text):
        covered += (float(x1) - float(x0)) * (float(y1) - float(y0))
    assert covered == pytest.approx(400.0 * 400.0)
