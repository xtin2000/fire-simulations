from __future__ import annotations

import numpy as np

from rxcadre_sim.annaburroo import head_speed


def test_head_speed_uses_first_and_last_observation():
    times = np.array([27.0, 53.0, 100.0])
    positions = np.array([15.0, 42.0, 100.0])
    assert round(head_speed(times, positions), 2) == 1.16
