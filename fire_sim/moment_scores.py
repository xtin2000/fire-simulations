"""The real and simulated fire compared at one moment, cell by cell and direction by direction.

At each fire's moment (C064 at 53 s, F19 at 86 s, about halfway across the plot) every cell
is -1 not burned, 0 burning or +1 burned, in the real fire and in each engine. The real fire's
arrival times are interpolated between NIST's measured outlines; a cell burns for BURNING_S.

Outputs in runs/moment/:
  confusion.csv          3x3 counts per fire and engine (rows real, columns simulated)
  scores.csv             outcome counts, precision, detection and Dice per state, fire Dice
  directions.csv         simulated minus real spread speed from the lighting line, per bearing
  <fire>_outcomes.png    map of hit, miss, false alarm and neither, one panel per engine
  <fire>_directions.png  the speed difference against compass bearing, one line per engine

Command:
  run   write all of the above for C064 and F19
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from fire_sim import elmfire_csiro as elm
from fire_sim import engine_output
from fire_sim import validation_metrics as vm
from fire_sim.elmfire_csiro import CsiroFire
from fire_sim.engine_output import FireRun
from fire_sim.paths import RUNS_DIR

MOMENT_DIR = RUNS_DIR / "moment"
MOMENT_S = {"C064": 53.0, "F19": 86.0}
LIGHTING_LINE_BAND_DEG = 20

OUTCOME_COLOURS = {
    vm.Outcome.NEITHER: "#e7e5df",
    vm.Outcome.HIT: "#2a78d6",
    vm.Outcome.MISS: "#eb6834",
    vm.Outcome.FALSE_ALARM: "#1baf7a",
}
OUTCOME_LABELS = {
    vm.Outcome.NEITHER: "neither",
    vm.Outcome.HIT: "hit",
    vm.Outcome.MISS: "miss (simulator behind)",
    vm.Outcome.FALSE_ALARM: "false alarm (simulator ahead)",
}
ENGINE_COLOURS = {
    engine_output.ELMFIRE: "#2a78d6",
    engine_output.WRF_SFIRE: "#eb6834",
    engine_output.FDS_FULL: "#1baf7a",
}


def lighting_line(fire: CsiroFire, run: FireRun) -> np.ndarray:
    xx, yy = np.meshgrid(run.x_m, run.y_m)
    on_line = np.abs(xx - fire.lighting_x_m) <= run.cell_m / 2
    return on_line & (np.abs(yy - fire.lighting_centre_y_m) <= fire.field_lighting_half_length_m)


def real_states(fire: CsiroFire, run: FireRun, time_s: float) -> np.ndarray:
    """The measured fire's cell states on the run's grid."""
    areas = vm.real_burned_areas(run.x_m, run.y_m, vm.measured_outlines(fire))
    arrival = vm.real_arrival_times(areas, lighting_line(fire, run))
    return vm.cell_states(arrival, time_s, vm.BURNING_S)


def off_plot_not_burned(states: np.ndarray, run: FireRun) -> np.ndarray:
    return np.where(run.on_plot(), states, vm.CellState.NOT_BURNED)


def score_row(fire: CsiroFire, run: FireRun, real: np.ndarray, simulated: np.ndarray) -> dict:
    matrix = vm.confusion_matrix(real, simulated)
    row = {"fire": fire.name, "engine": run.engine, "time_s": MOMENT_S[fire.name]}
    row |= vm.outcome_counts(vm.outcomes(real, simulated))
    for state in vm.FIRE_STATES:
        scores = vm.state_scores(matrix, state)
        name = state.name.lower()
        row |= {f"{name}_{key}": value for key, value in scores.__dict__.items()}
    return row | {"fire_dice": vm.mean_fire_dice(matrix)}


def confusion_rows(fire: CsiroFire, run: FireRun, matrix: np.ndarray) -> list[dict]:
    return [
        {"fire": fire.name, "engine": run.engine, "real": real_state.name.lower()}
        | {sim_state.name.lower(): int(matrix[i, j]) for j, sim_state in enumerate(vm.CellState)}
        for i, real_state in enumerate(vm.CellState)
    ]


def plot_outcomes(fire: CsiroFire, maps: dict[str, tuple[FireRun, np.ndarray]]) -> None:
    colours = ListedColormap([OUTCOME_COLOURS[outcome] for outcome in vm.Outcome])
    figure, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.6), sharey=True)
    for axis, (engine, (run, cell_outcomes)) in zip(np.atleast_1d(axes), maps.items(), strict=True):
        extent = (run.x_m[0], run.x_m[-1], run.y_m[0], run.y_m[-1])
        axis.imshow(
            cell_outcomes,
            origin="lower",
            extent=extent,
            cmap=colours,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis.set_xlim(-5, fire.plot_length_m + 5)
        axis.set_ylim(-fire.plot_half_width_m - 5, fire.plot_half_width_m + 5)
        axis.set_title(engine, fontsize=12)
        axis.set_xlabel("x along the wind (m)")
    np.atleast_1d(axes)[0].set_ylabel("y (m)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLOURS[outcome], label=OUTCOME_LABELS[outcome])
        for outcome in (vm.Outcome.HIT, vm.Outcome.MISS, vm.Outcome.FALSE_ALARM, vm.Outcome.NEITHER)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(f"{fire.name} at {MOMENT_S[fire.name]:.0f} s", fontsize=13)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(MOMENT_DIR / f"{fire.name}_outcomes.png", dpi=150)
    plt.close(figure)


def plot_directions(fire: CsiroFire, differences: dict[str, np.ndarray]) -> None:
    bearings = vm.direction_bearings_deg()
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.axhline(0, color="#52514e", linewidth=1)
    # The lighting line runs north-south through the origin, so near 180 and 360 deg the reach
    # is how far each fire was lit along the line, not how far it spread.
    for low, high in [(180, 180 + LIGHTING_LINE_BAND_DEG), (360 - LIGHTING_LINE_BAND_DEG, 360)]:
        axis.axvspan(low, high, color="#e7e5df", linewidth=0)
    for engine, difference in differences.items():
        axis.plot(bearings, difference, color=ENGINE_COLOURS[engine], linewidth=2, label=engine)
    axis.axvline(elm.travel_bearing(fire.name), color="#52514e", linewidth=1, linestyle=":")
    axis.set_xlim(180, 360)
    axis.set_xticks(range(180, 361, 30))
    axis.set_xlabel(
        "Compass bearing from the lighting line's centre (deg)\n"
        "dotted: real head's direction; grey: along the lighting line"
    )
    axis.set_ylabel("Simulated − real speed (m/s)")
    axis.set_title(f"{fire.name} at {MOMENT_S[fire.name]:.0f} s: + simulator faster, − slower")
    axis.grid(axis="y", color="#e7e5df")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(MOMENT_DIR / f"{fire.name}_directions.png", dpi=150)
    plt.close(figure)


def run() -> None:
    """Score every engine at each fire's moment and write the tables and pictures."""
    MOMENT_DIR.mkdir(parents=True, exist_ok=True)
    scores, confusion, directions = [], [], []
    for fire in elm.FIRES:
        time_s = MOMENT_S[fire.name]
        maps, differences = {}, {}
        for fire_run in engine_output.runs_for(fire):
            real = off_plot_not_burned(real_states(fire, fire_run, time_s), fire_run)
            simulated = vm.cell_states(fire_run.arrival_s, time_s, vm.BURNING_S)
            simulated = off_plot_not_burned(simulated, fire_run)
            scores.append(score_row(fire, fire_run, real, simulated))
            confusion += confusion_rows(fire, fire_run, vm.confusion_matrix(real, simulated))
            maps[fire_run.engine] = (fire_run, vm.outcomes(real, simulated))
            origin = (fire.lighting_x_m, fire.lighting_centre_y_m)
            difference = vm.speed_difference_by_direction(
                vm.fire_reached(real),
                vm.fire_reached(simulated),
                fire_run.x_m,
                fire_run.y_m,
                origin=origin,
                time_s=time_s,
            )
            differences[fire_run.engine] = difference
            directions += [
                {"fire": fire.name, "engine": fire_run.engine, "bearing_deg": b, "m_s": d}
                for b, d in zip(vm.direction_bearings_deg(), difference, strict=True)
            ]
        plot_outcomes(fire, maps)
        plot_directions(fire, differences)
    for name, rows in [("scores", scores), ("confusion", confusion), ("directions", directions)]:
        pd.DataFrame(rows).to_csv(MOMENT_DIR / f"{name}.csv", index=False)
    print(pd.DataFrame(scores).round(2).to_string(index=False))
    print(f"Outputs: {MOMENT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["run"])
    {"run": run}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
