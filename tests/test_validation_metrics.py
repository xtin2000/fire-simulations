from __future__ import annotations

import numpy as np
import pytest

from fire_sim import engine_output
from fire_sim import validation_metrics as vm


def square(rows: slice, columns: slice, size: int = 20) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[rows, columns] = True
    return mask


def test_identical_shapes_score_perfectly() -> None:
    mask = square(slice(5, 15), slice(5, 15))
    scores = vm.shape_scores(mask, mask, cell_m=1.0)
    assert (scores.dice, scores.jaccard, scores.false_alarm_ratio) == (1.0, 1.0, 0.0)


def test_disjoint_shapes_have_no_overlap() -> None:
    scores = vm.shape_scores(
        square(slice(0, 5), slice(0, 5)), square(slice(10, 15), slice(10, 15)), 1.0
    )
    assert scores.dice == 0.0


def test_half_overlap_gives_known_dice_and_detection() -> None:
    predicted = square(slice(0, 10), slice(0, 10))
    observed = square(slice(0, 10), slice(5, 15))
    scores = vm.shape_scores(predicted, observed, cell_m=1.0)
    assert (scores.dice, scores.probability_of_detection, scores.false_alarm_ratio) == (
        0.5,
        0.5,
        0.5,
    )


def test_shifted_square_boundary_distance_is_the_shift_in_metres() -> None:
    predicted = square(slice(5, 10), slice(5, 10), size=40)
    observed = square(slice(5, 10), slice(8, 13), size=40)
    _, worst = vm.boundary_distances(predicted, observed, cell_m=2.0)
    assert worst == pytest.approx(6.0)


def test_time_to_reach_interpolates_between_samples() -> None:
    assert (
        engine_output.time_to_reach(np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0, 30.0]), 20.0)
        == 15.0
    )


def test_time_to_reach_is_nan_when_target_is_never_reached() -> None:
    assert np.isnan(engine_output.time_to_reach(np.array([0.0, 10.0]), np.array([0.0, 5.0]), 8.0))


def test_outline_mask_fills_the_closed_outline() -> None:
    x = y = np.arange(10) + 0.5
    outline = np.array([[2.0, 2.0], [2.0, 8.0], [8.0, 8.0]])
    assert vm.outline_mask(x, y, outline)[6, 3]


@pytest.fixture
def meeting_example_matrix() -> np.ndarray:
    """The 4x4 grid example from the 17 Nov meeting, as (real, simulated) state pairs."""
    pairs = [(-1, -1)] * 11 + [(-1, 1), (0, -1), (0, 0), (1, 0), (1, 1)]
    real, simulated = (np.array(states).reshape(4, 4) for states in zip(*pairs, strict=True))
    return vm.confusion_matrix(real, simulated)


@pytest.fixture
def spreading_fire_arrival_s() -> np.ndarray:
    """A fire crossing a 20 x 20 plot west to east at 1 cell per second."""
    return np.tile(np.arange(20, dtype=float), (20, 1))


def test_confusion_matrix_matches_the_meeting_example(meeting_example_matrix: np.ndarray) -> None:
    assert meeting_example_matrix.tolist() == [[11, 0, 1], [1, 1, 0], [0, 1, 1]]


def test_state_scores_of_the_meeting_example(meeting_example_matrix: np.ndarray) -> None:
    scores = [vm.state_scores(meeting_example_matrix, state) for state in vm.CellState]
    assert [(s.precision, s.detection) for s in scores] == [
        (11 / 12, 11 / 12),
        (0.5, 0.5),
        (0.5, 0.5),
    ]


def test_mean_fire_dice_leaves_out_not_burned(meeting_example_matrix: np.ndarray) -> None:
    assert vm.mean_fire_dice(meeting_example_matrix) == 0.5


def test_cell_states_split_burning_from_burned(spreading_fire_arrival_s: np.ndarray) -> None:
    states = vm.cell_states(spreading_fire_arrival_s, time_s=10.0, burning_s=4.0)
    assert states[0, [6, 7, 10, 11]].tolist() == [1, 0, 0, -1]


def test_never_reached_cells_stay_not_burned() -> None:
    assert vm.cell_states(np.array([np.nan]), time_s=100.0, burning_s=5.0)[0] == -1


def test_identical_fire_has_only_diagonal_counts(spreading_fire_arrival_s: np.ndarray) -> None:
    states = vm.cell_states(spreading_fire_arrival_s, time_s=10.0, burning_s=4.0)
    matrix = vm.confusion_matrix(states, states)
    assert np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0


def test_too_fast_fire_puts_real_burning_cells_in_simulated_burned(
    spreading_fire_arrival_s: np.ndarray,
) -> None:
    real = vm.cell_states(spreading_fire_arrival_s, time_s=10.0, burning_s=4.0)
    too_fast = vm.cell_states(spreading_fire_arrival_s / 2, time_s=10.0, burning_s=4.0)
    burning_row = vm.confusion_matrix(real, too_fast)[1]
    assert burning_row[2] > 0


def test_signed_gap_is_positive_when_the_simulated_fire_is_too_big() -> None:
    real = square(slice(10, 20), slice(10, 20), size=40)
    too_big = square(slice(8, 22), slice(8, 22), size=40)
    assert vm.signed_boundary_distance(too_big, real, cell_m=1.0) > 0


def test_signed_gap_is_negative_when_the_simulated_fire_is_too_small() -> None:
    real = square(slice(10, 20), slice(10, 20), size=40)
    too_small = square(slice(12, 18), slice(12, 18), size=40)
    assert vm.signed_boundary_distance(too_small, real, cell_m=1.0) < 0


def test_shifted_fire_cancels_in_signed_gap_but_not_in_outline_gap() -> None:
    real = square(slice(10, 20), slice(10, 20), size=40)
    shifted = square(slice(10, 20), slice(14, 24), size=40)
    unsigned_gap, _ = vm.boundary_distances(shifted, real, cell_m=1.0)
    signed_gap = vm.signed_boundary_distance(shifted, real, cell_m=1.0)
    assert abs(signed_gap) < unsigned_gap / 2


def test_real_burned_area_keeps_ground_inside_earlier_outlines() -> None:
    x = y = np.arange(10) + 0.5
    early = np.array([[0.0, 0.0], [0.0, 10.0], [3.0, 10.0], [3.0, 0.0]])
    late_starting_away_from_line = np.array([[5.0, 0.0], [5.0, 10.0], [8.0, 10.0], [8.0, 0.0]])
    areas = vm.real_burned_areas(x, y, {10.0: early, 20.0: late_starting_away_from_line})
    assert areas[20.0][5, 1]


@pytest.mark.parametrize(
    ("real", "simulated", "expected"),
    [
        (-1, -1, vm.Outcome.NEITHER),
        (-1, 0, vm.Outcome.FALSE_ALARM),
        (-1, 1, vm.Outcome.FALSE_ALARM),
        (0, -1, vm.Outcome.MISS),
        (0, 0, vm.Outcome.HIT),
        (0, 1, vm.Outcome.FALSE_ALARM),
        (1, -1, vm.Outcome.MISS),
        (1, 0, vm.Outcome.MISS),
        (1, 1, vm.Outcome.HIT),
    ],
)
def test_each_state_pair_has_the_agreed_outcome(
    real: int, simulated: int, expected: vm.Outcome
) -> None:
    assert vm.outcomes(np.array([real]), np.array([simulated]))[0] == expected


def test_reach_by_direction_finds_the_farthest_burned_cell_each_way() -> None:
    x = y = np.arange(-10, 11, dtype=float)
    burned = np.zeros((y.size, x.size), dtype=bool)
    burned[10, 10:18] = True
    reach = vm.reach_by_direction(burned, x, y, origin=(0.0, 0.0))
    west_sector = int(270 / vm.DIRECTION_BIN_DEG)
    assert (reach[west_sector], reach.sum()) == (7.0, 7.0)


def test_speed_difference_is_positive_where_the_simulator_spreads_farther() -> None:
    x = y = np.arange(-10, 11, dtype=float)
    real, simulated = (np.zeros((y.size, x.size), dtype=bool) for _ in range(2))
    real[10, 10:15] = True
    simulated[10, 10:19] = True
    difference = vm.speed_difference_by_direction(
        real, simulated, x, y, origin=(0.0, 0.0), time_s=2.0
    )
    assert difference.max() == 2.0


def test_real_arrival_is_interpolated_between_outlines() -> None:
    lit = square(slice(None), slice(0, 1))
    areas = {10.0: square(slice(None), slice(0, 5)), 20.0: square(slice(None), slice(0, 15))}
    arrival = vm.real_arrival_times(areas, lit)
    assert arrival[0, [0, 4, 9, 14]] == pytest.approx([0.0, 10.0, 15.0, 20.0], abs=1.5)
