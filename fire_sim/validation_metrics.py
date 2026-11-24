"""Scores that compare a simulated fire with the measured one, the same way for every engine.

Two kinds of score:

* Shape scores compare the burnt area at one time with CSIRO's measured outline: Dice and Jaccard
  (overlap), probability of detection and false alarm ratio (missed and invented fire), and the
  mean and worst distance between the two outlines. They exist only for C064 and F19, the two
  fires NIST measured outlines for.
* State scores keep three states per cell (-1 not burned, 0 burning, +1 burned) and count every
  real/simulated pair in a 3x3 confusion matrix, giving precision, detection and Dice per state.
  Each pair is also a hit, miss (simulator behind), false alarm (simulator ahead) or neither,
  and the reach of both fires from the lighting line is compared direction by direction.
* Timing scores compare when the head reaches 80 % of the mapped distance. They exist for all
  eight fires, because CSIRO's burn table gives the head position for each of them.

Commands:
  shape   C064 and F19: shape scores of ELMFIRE, WRF-SFIRE and FDS full physics at the outline times
  timing  all eight fires: time to 80 % of the mapped distance, all three engines
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
from matplotlib.path import Path as OutlinePath
from scipy import ndimage

from fire_sim import elmfire_csiro as elm
from fire_sim import engine_output
from fire_sim.elmfire_csiro import Conditions, CsiroFire, Fuel
from fire_sim.paths import CSIRO_RAW_DIR, RUNS_DIR

# Measured outlines are only a closed burnt area while both ends still touch the lighting line.
# Later ones show just the flame zone, which closing with a straight line would misrepresent.
MAX_OUTLINE_END_X_M = 40.0


# Flame depth over spread rate in CSIRO's burn table: 5.6 m / 1.18 m/s = 4.7 s for C064 at 53 s,
# 9.5 m / 1.52 m/s = 6.2 s for F19 at 86 s. One value for both fires and all engines.
BURNING_S = 5.0
DIRECTION_BIN_DEG = 5.0


class CellState(IntEnum):
    NOT_BURNED = -1
    BURNING = 0
    BURNED = 1


FIRE_STATES = (CellState.BURNING, CellState.BURNED)


class Outcome(IntEnum):
    NEITHER = 0
    HIT = 1
    MISS = 2
    FALSE_ALARM = 3


@dataclass(frozen=True)
class StateScores:
    precision: float
    detection: float
    dice: float


@dataclass(frozen=True)
class ShapeScores:
    dice: float
    jaccard: float
    probability_of_detection: float
    precision: float
    false_alarm_ratio: float
    mean_boundary_distance_m: float
    hausdorff_m: float
    signed_boundary_distance_m: float


def overlap_counts(predicted: np.ndarray, observed: np.ndarray) -> tuple[int, int, int]:
    """Cells burned in both, only in the prediction, and only in the observation."""
    both = int(np.sum(predicted & observed))
    return both, int(np.sum(predicted & ~observed)), int(np.sum(~predicted & observed))


def boundary_cells(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def boundary_distances(
    predicted: np.ndarray, observed: np.ndarray, cell_m: float
) -> tuple[float, float]:
    """Mean and worst distance (m) between the two outlines, counted both ways round."""
    predicted_edge, observed_edge = boundary_cells(predicted), boundary_cells(observed)
    to_observed = ndimage.distance_transform_edt(~observed_edge)[predicted_edge] * cell_m
    to_predicted = ndimage.distance_transform_edt(~predicted_edge)[observed_edge] * cell_m
    mean = (to_observed.mean() + to_predicted.mean()) / 2
    return float(mean), float(max(to_observed.max(), to_predicted.max()))


def shape_scores(predicted: np.ndarray, observed: np.ndarray, cell_m: float) -> ShapeScores:
    both, invented, missed = overlap_counts(predicted, observed)
    mean_distance, worst_distance = boundary_distances(predicted, observed, cell_m)
    return ShapeScores(
        dice=2 * both / (2 * both + invented + missed),
        jaccard=both / (both + invented + missed),
        probability_of_detection=both / (both + missed),
        precision=both / (both + invented) if both + invented else 0.0,
        false_alarm_ratio=invented / (both + invented) if both + invented else 1.0,
        mean_boundary_distance_m=mean_distance,
        hausdorff_m=worst_distance,
        signed_boundary_distance_m=signed_boundary_distance(predicted, observed, cell_m),
    )


def cell_states(arrival_s: np.ndarray, time_s: float, burning_s: float) -> np.ndarray:
    """State of every cell at one time, from arrival times (NaN = never reached)."""
    states = np.full(arrival_s.shape, CellState.NOT_BURNED, dtype=np.int8)
    states[arrival_s <= time_s] = CellState.BURNING
    states[arrival_s + burning_s <= time_s] = CellState.BURNED
    return states


def fire_reached(states: np.ndarray) -> np.ndarray:
    """Burning or burned: the binary view that Dice and the outline gap use."""
    return states >= CellState.BURNING


def confusion_matrix(real: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """Cell counts with rows = real state and columns = simulated state, in CellState order."""
    pair_index = (real.astype(int) + 1) * 3 + (simulated.astype(int) + 1)
    return np.bincount(pair_index.ravel(), minlength=9).reshape(3, 3)


def state_scores(matrix: np.ndarray, state: CellState) -> StateScores:
    """One state against the other two: precision down its column, detection along its row."""
    index = state + 1
    hits = matrix[index, index]
    simulated_total, real_total = matrix[:, index].sum(), matrix[index, :].sum()
    precision = hits / simulated_total if simulated_total else 0.0
    detection = hits / real_total if real_total else 0.0
    dice = 2 * hits / (simulated_total + real_total) if simulated_total + real_total else 0.0
    return StateScores(float(precision), float(detection), float(dice))


def mean_fire_dice(matrix: np.ndarray) -> float:
    """Dice averaged over burning and burned; not burned is left out, as in the binary Dice."""
    return float(np.mean([state_scores(matrix, state).dice for state in FIRE_STATES]))


def outcomes(real: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """Each cell's Outcome. States are ordered by how far the fire has got, so a simulated
    state above the real one is the simulator ahead (false alarm), below it behind (miss)."""
    result = np.full(real.shape, Outcome.NEITHER, dtype=np.int8)
    result[simulated > real] = Outcome.FALSE_ALARM
    result[simulated < real] = Outcome.MISS
    result[(simulated == real) & (real != CellState.NOT_BURNED)] = Outcome.HIT
    return result


def outcome_counts(cell_outcomes: np.ndarray) -> dict[str, int]:
    counts = np.bincount(cell_outcomes.ravel(), minlength=len(Outcome))
    return {outcome.name.lower(): int(counts[outcome]) for outcome in Outcome}


def compass_bearings_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass bearing of each offset; the simulation frame is the real plot turned half a turn."""
    return (np.degrees(np.arctan2(dx, dy)) + 180.0) % 360.0


def reach_by_direction(
    burned: np.ndarray, x: np.ndarray, y: np.ndarray, origin: tuple[float, float]
) -> np.ndarray:
    """Farthest burned cell from origin (m) in each DIRECTION_BIN_DEG compass sector, 0 if none."""
    xx, yy = np.meshgrid(x, y)
    dx, dy = xx[burned] - origin[0], yy[burned] - origin[1]
    sector = (compass_bearings_deg(dx, dy) // DIRECTION_BIN_DEG).astype(int)
    reach = np.zeros(int(360 / DIRECTION_BIN_DEG))
    np.maximum.at(reach, sector, np.hypot(dx, dy))
    return reach


def direction_bearings_deg() -> np.ndarray:
    """Centre of each compass sector that reach_by_direction uses."""
    return np.arange(0.0, 360.0, DIRECTION_BIN_DEG) + DIRECTION_BIN_DEG / 2


def speed_difference_by_direction(
    real_burned: np.ndarray,
    simulated_burned: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    origin: tuple[float, float],
    time_s: float,
) -> np.ndarray:
    """Simulated minus real average spread speed (m/s) from origin in each compass sector:
    + the simulator is faster that way, - slower. A difference, not a ratio, so a direction
    only one fire reached stays finite."""
    simulated_reach = reach_by_direction(simulated_burned, x, y, origin)
    return (simulated_reach - reach_by_direction(real_burned, x, y, origin)) / time_s


def signed_boundary_distance(predicted: np.ndarray, observed: np.ndarray, cell_m: float) -> float:
    """Mean distance (m) between the outlines, + where the simulated fire reaches past the real
    edge and - where it falls short, counted both ways round like the outline gap.

    Reported beside the unsigned outline gap, never instead of it: a fire that is the right
    size but shifted is half ahead and half behind, so its signed mean cancels to about 0.
    """
    predicted_edge, observed_edge = boundary_cells(predicted), boundary_cells(observed)
    to_observed = ndimage.distance_transform_edt(~observed_edge)[predicted_edge] * cell_m
    to_predicted = ndimage.distance_transform_edt(~predicted_edge)[observed_edge] * cell_m
    predicted_sign = np.where(observed[predicted_edge], -1.0, 1.0)
    observed_sign = np.where(predicted[observed_edge], 1.0, -1.0)
    return float(
        (np.mean(predicted_sign * to_observed) + np.mean(observed_sign * to_predicted)) / 2
    )


def outline_mask(x: np.ndarray, y: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Cells inside the measured outline, closed by a straight line between its two ends."""
    xx, yy = np.meshgrid(x, y)
    inside = OutlinePath(outline).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    return inside.reshape(xx.shape)


def measured_outlines(fire: CsiroFire) -> dict[float, np.ndarray]:
    """Measured outlines by time (s) as [x, y] points; flame-zone-only ones are left out."""
    table = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_flame_position_data.csv")
    outlines = {}
    for time_s in sorted({float(column.split()[0]) for column in table.columns}):
        points = table[[f"{time_s:.0f} x", f"{time_s:.0f} y"]].dropna().to_numpy()
        if max(points[0, 0], points[-1, 0]) <= MAX_OUTLINE_END_X_M:
            outlines[time_s] = points
    return outlines


def real_burned_areas(
    x: np.ndarray, y: np.ndarray, outlines: dict[float, np.ndarray]
) -> dict[float, np.ndarray]:
    """Burned area at each outline time, including everything inside earlier outlines.

    A later outline can start away from the lighting line, so its own closed area would leave
    out ground the fire had already burned and charge the engines with false alarms there.
    """
    areas, burned = {}, np.zeros((y.size, x.size), dtype=bool)
    for time_s, outline in sorted(outlines.items()):
        burned = burned | outline_mask(x, y, outline)
        areas[time_s] = burned
    return areas


def real_arrival_times(burned_areas: dict[float, np.ndarray], lit: np.ndarray) -> np.ndarray:
    """Arrival time (s) of the real fire in each cell, NaN past the last outline.

    Between two outlines the time is interpolated by distance: a cell a third of the way from
    the earlier edge to the later one gets a third of the time between them. Before the first
    outline the earlier edge is the lighting line at time 0, which ignores the minute or so
    the lighting took; scoring after the first outline does not depend on it.
    """
    arrival = np.full(lit.shape, np.nan)
    arrival[lit] = 0.0
    earlier_time, earlier = 0.0, lit
    for time_s, burned in sorted(burned_areas.items()):
        ring = burned & ~earlier
        # Both edges lie half a cell beyond the centres of the cells either side of them.
        from_earlier = ndimage.distance_transform_edt(~earlier)[ring] - 0.5
        to_later = ndimage.distance_transform_edt(burned)[ring] - 0.5
        share = from_earlier / (from_earlier + to_later)
        arrival[ring] = earlier_time + share * (time_s - earlier_time)
        earlier_time, earlier = time_s, earlier | burned
    return arrival


def shape() -> None:
    """Shape scores at each measured outline time for C064 and F19."""
    rows = []
    for fire in elm.FIRES:
        outlines = measured_outlines(fire)
        for run in engine_output.runs_for(fire):
            on_plot = run.on_plot()
            for time_s, real_area in real_burned_areas(run.x_m, run.y_m, outlines).items():
                observed = real_area & on_plot
                predicted = run.burned_at(time_s) & on_plot
                rows.append(
                    {"fire": fire.name, "engine": run.engine, "time_s": time_s}
                    | shape_scores(predicted, observed, run.cell_m).__dict__
                )
    table = pd.DataFrame(rows)
    path = RUNS_DIR / "shape_scores.csv"
    table.to_csv(path, index=False)
    print(table.round(2).to_string(index=False))
    print(f"Table: {path}")


def timing() -> None:
    """Time to 80 % of the mapped distance for all eight fires, all three engines."""
    rows = []
    for fire in elm.all_fires():
        target_m, real_s = engine_output.real_time_to_fraction(fire.name)
        row = {"fire": fire.name, "target_m": round(target_m), "real_s": round(real_s)}
        runs = [
            engine_output.from_elmfire(fire, Fuel.PATCHWORK, Conditions.FIELD_STATION),
            engine_output.from_wrf_sfire(fire, Fuel.PATCHWORK, Conditions.FIELD_STATION),
            engine_output.from_fds_full(fire, turns_with_station=True),
        ]
        for run in filter(None, runs):
            reached_s = engine_output.engine_time_to_fraction(run)
            row[f"{run.engine}_s"] = round(reached_s) if np.isfinite(reached_s) else None
            row[f"{run.engine}_vs_real"] = round(reached_s / real_s, 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    path = RUNS_DIR / "timing_scores.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"Table: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["shape", "timing"])
    {"shape": shape, "timing": timing}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
