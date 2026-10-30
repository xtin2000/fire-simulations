from __future__ import annotations

import numpy as np
import pytest

from rxcadre_sim.csiro_fds_fast import NIST_INPUTS_DIR, fast_input, head_positions
from rxcadre_sim.elmfire_csiro import FIRES, Fuel

C064 = FIRES[0]
needs_nist_inputs = pytest.mark.skipif(
    not (NIST_INPUTS_DIR / "Case_C064_LS.fds").exists(),
    reason="run `python -m rxcadre_sim.csiro_fds_fast download` first",
)


@needs_nist_inputs
def test_fast_input_uses_mode_1_on_a_1_m_grid():
    text = fast_input(C064, Fuel.STANDARD)
    assert "LEVEL_SET_MODE=1" in text and "IJK=120,120,6" in text


@needs_nist_inputs
def test_measured_grass_sets_load_and_height():
    text = fast_input(C064, Fuel.MEASURED)
    assert "VEG_LSET_SURF_LOAD = 0.283, VEG_LSET_HT = 0.21" in text


@needs_nist_inputs
def test_fast_input_drops_nists_output_devices():
    assert "&DEVC" not in fast_input(C064, Fuel.STANDARD)


def test_head_position_reads_the_strip_south_of_the_centre_line():
    x = np.arange(10.0)
    y = np.array([-2.0, 2.0])
    arrival = np.array([np.arange(10.0), np.zeros(10)])
    assert head_positions(x, y, arrival, np.array([4.0]))[0] == 4.0
