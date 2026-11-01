from __future__ import annotations

import numpy as np
import pytest

from fire_sim.annaburroo import (
    ADDED_FIRE_NAMES,
    FIRE_NAMES,
    PlotGeometry,
    fuel_samples,
    head_speed,
    plot_geometry,
)


def test_head_speed_uses_first_and_last_observation():
    times = np.array([27.0, 53.0, 100.0])
    positions = np.array([15.0, 42.0, 100.0])
    assert round(head_speed(times, positions), 2) == 1.16


def test_straight_lighting_line_runs_north_south_around_its_centre():
    plot = PlotGeometry(100, 100, -2.5, -7, 50, 0)
    assert plot.lighting_ends() == ((-2.5, -32.0), (-2.5, 18.0))


def test_tilted_lighting_line_keeps_its_length_and_moves_west_going_south():
    (x_north, y_north), (x_south, y_south) = PlotGeometry(100, 100, 6, -24, 50, 30).lighting_ends()
    assert np.hypot(x_south - x_north, y_south - y_north) == pytest.approx(50)
    assert x_south > x_north
    assert y_south > y_north


@pytest.mark.parametrize("fire_name", ADDED_FIRE_NAMES)
def test_fuel_samples_lie_inside_their_plot(fire_name):
    plot = plot_geometry(fire_name)
    samples = fuel_samples(fire_name)
    assert samples.x_m.between(0, plot.east_west_m).all()
    assert (samples.y_m.abs() <= plot.north_south_m / 2).all()


@pytest.mark.parametrize("fire_name", (*FIRE_NAMES, *ADDED_FIRE_NAMES))
def test_every_fuel_sample_has_load_and_height(fire_name):
    samples = fuel_samples(fire_name)
    assert samples[["load_kg_m2", "height_m"]].notna().all(axis=None)
    assert samples.load_kg_m2.gt(0).all()


def test_e11_keeps_the_fifteen_samples_left_after_the_struck_out_load():
    assert len(fuel_samples("E11")) == 15
