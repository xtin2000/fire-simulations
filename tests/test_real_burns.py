from __future__ import annotations

import numpy as np

from fire_sim.real_burns import CELL_SIZE_M, MAX_FILL_DISTANCE_M, Grid, fill_gaps, grid_for


def test_grid_covers_the_block_with_a_margin():
    block = np.array([[1000.0, 2000.0], [1800.0, 2000.0], [1800.0, 2600.0], [1000.0, 2600.0]])
    grid = grid_for(block)
    assert grid.origin_x < 1000.0 and grid.origin_y < 2000.0
    assert grid.origin_x + grid.cols * CELL_SIZE_M > 1800.0
    assert grid.origin_y + grid.rows * CELL_SIZE_M > 2600.0


def test_cell_lookup_flags_points_outside_the_grid():
    grid = Grid(0.0, 0.0, rows=10, cols=10)
    _, _, inside = grid.cell_of(np.array([4.0, 200.0]), np.array([4.0, 4.0]))
    assert inside.tolist() == [True, False]


def test_gap_filling_only_reaches_cells_near_a_detection():
    arrival = np.full((1, 20), np.nan)
    arrival[0, 0] = 100.0
    filled = fill_gaps(arrival, np.ones_like(arrival, dtype=bool))
    reach = int(MAX_FILL_DISTANCE_M // CELL_SIZE_M)
    assert np.all(filled[0, : reach + 1] == 100.0)
    assert np.isnan(filled[0, reach + 1])
