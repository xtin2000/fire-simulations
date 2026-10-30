from __future__ import annotations

from fire_sim.ros_table import TableCase, all_cases, build_input


def test_table_covers_every_axis_combination():
    assert len(all_cases()) == 9 * 3 * 7 * 6


def test_case_ids_are_unique_and_valid_fds_chids():
    chids = [case.chid for case in all_cases()]
    assert len(set(chids)) == len(chids)
    assert all("." not in chid for chid in chids)


def test_front_is_tracked_from_a_slice_not_arrival_time_devices():
    """FIRE ARRIVAL TIME devices written only at the end of a run report bogus times."""
    text = build_input(TableCase(3.0, 0.08, 0.4, 0.3))
    assert "QUANTITY='LEVEL SET VALUE'" in text
    assert "FIRE ARRIVAL TIME" not in text
