from __future__ import annotations

import re

import pytest

from fire_sim import annaburroo
from fire_sim import csiro_fds_added as added
from fire_sim.csiro_fds_full import FULL_INPUTS_DIR
from fire_sim.paths import CSIRO_RAW_DIR

needs_inputs = pytest.mark.skipif(
    not (FULL_INPUTS_DIR / "Case_C064_crude.fds").exists()
    or not any(CSIRO_RAW_DIR.rglob("Plot_Average_Data.csv")),
    reason="run the csiro_fds_full and annaburroo download commands first",
)


def case_input(fire_name: str) -> str:
    case = next(c for c in added.added_cases() if c.fire.name == fire_name)
    return added.added_input(case)


def ignition_vents(text: str) -> list[str]:
    return re.findall(r"&VENT [^\n]*SURF_ID='IGNITION'[^\n]*", text)


@needs_inputs
def test_each_added_fire_gets_one_case() -> None:
    assert [c.fire.name for c in added.added_cases()] == list(annaburroo.ADDED_FIRE_NAMES)


@needs_inputs
def test_straight_lighting_line_is_one_strip_on_the_lines_centre() -> None:
    (vent,) = ignition_vents(case_input("C062"))
    assert "XB=0.00,1.60,-32.00,18.00" in vent and "XYZ=0,-7,0" in vent


@needs_inputs
def test_tilted_lighting_line_is_a_staircase_that_lights_from_its_centre() -> None:
    vents = ignition_vents(case_input("C112"))
    assert len(vents) > 20 and all("XYZ=6,-24,0" in v for v in vents)


@needs_inputs
def test_hundred_metre_plots_use_the_small_domain() -> None:
    assert "XB=90,110,40,60" in case_input("C062")


@needs_inputs
def test_two_hundred_metre_plots_use_the_large_domain() -> None:
    assert "XB=180,220,80,120" in case_input("F05")


@needs_inputs
def test_roads_around_the_plot_are_left_out() -> None:
    assert "SURF_ID='ROAD'" not in re.sub(r"&SURF ID='ROAD'[^/]*/", "", case_input("E11"))


@needs_inputs
def test_run_length_is_the_fires_own() -> None:
    case = next(c for c in added.added_cases() if c.fire.name == "C112")
    assert f"T_END={case.fire.end_time_s:g}" in added.added_input(case)
