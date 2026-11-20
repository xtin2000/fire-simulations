from __future__ import annotations

import numpy as np
import pytest

from fire_sim import elmfire_csiro as elm
from fire_sim import engine_output


@pytest.fixture
def run() -> engine_output.FireRun:
    fire = elm.FIRES[0]
    x, y = engine_output._grid_north_last(fire)
    arrival = np.where(np.meshgrid(x, y)[0] < 30, np.meshgrid(x, y)[0] + 10.0, np.nan)
    return engine_output.FireRun(engine_output.ELMFIRE, fire, "synthetic", x, y, arrival)


def test_common_grid_runs_south_to_north(run: engine_output.FireRun) -> None:
    assert run.y_m[0] < run.y_m[-1]


def test_burned_cells_are_those_reached_by_the_time(run: engine_output.FireRun) -> None:
    assert run.burned_at(20.0).sum() < run.burned_at(40.0).sum()


def test_burnt_area_counts_only_cells_on_the_plot(run: engine_output.FireRun) -> None:
    plot_area = run.fire.plot_length_m * 2 * run.fire.plot_half_width_m
    assert run.burnt_area_m2(1e9) < plot_area


def test_head_distance_is_nan_before_anything_burns(run: engine_output.FireRun) -> None:
    assert np.isnan(run.head_distance_m(np.array([0.0]), 270.0)[0])


def test_fds_domain_corner_is_the_lowest_mesh_corner() -> None:
    meshes = (
        "&MESH IJK=20,20,20, XB=-10,10,-60,-40,0,20, MPI_PROCESS=0 /\n"
        "&MESH IJK=20,20,20, XB=10,30,-40,-20,0,20, MPI_PROCESS=0 /\n"
    )
    assert engine_output.fds_domain_corner(meshes) == (-10.0, -60.0)
