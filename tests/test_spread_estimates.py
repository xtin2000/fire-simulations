from __future__ import annotations

import numpy as np
import pytest

from fire_sim.s5_fds_fast import spread_rate_from_arrivals
from fire_sim.scenario_export import spread_velocity


def test_plane_fit_recovers_rate_of_spread_from_sensor_arrivals():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 50, 6), rng.uniform(-80, -40, 6)
    arrival = 100.0 - y / 0.12
    assert spread_rate_from_arrivals(x, y, arrival) == pytest.approx(0.12)


def test_spread_velocity_points_down_the_arrival_gradient():
    columns = np.arange(200)
    arrival = np.broadcast_to(columns * 2.0 / 0.25, (200, 200)).astype(float)
    velocity = spread_velocity(arrival)
    interior = velocity[50:150, 50:150]
    assert interior[..., 0] == pytest.approx(0.25, rel=1e-3)
    assert interior[..., 1] == pytest.approx(0.0, abs=1e-6)


def test_spread_velocity_is_zero_where_fire_never_arrives():
    arrival = np.full((200, 200), np.nan)
    arrival[:, :100] = np.arange(100) * 8.0
    assert np.all(spread_velocity(arrival)[:, 100:] == 0)
