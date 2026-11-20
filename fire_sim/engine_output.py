"""One format for what ELMFIRE, WRF-SFIRE and FDS full physics each say about a CSIRO fire.

The engines write different things: ELMFIRE a time-of-arrival raster with north first, WRF-SFIRE
a NetCDF of ignition times on its own fire map, FDS a ground-temperature slice on its own 1 m
cells. Here each becomes a `FireRun`: the fire's arrival time on ELMFIRE's 1 m cell grid, in the
simulation frame (x along the wind, y increasing to the north), in seconds after lighting, NaN
where the fire never arrived. Everything else (heads, areas, shape scores)
is computed from that one array, so a difference between engines is never a difference in how
their output was read.

How each engine decides a cell has burned is its own, and is kept in `burned_means`.

Commands:
  summary  C064 and F19: the three engines side by side (area burnt, head position, time to 80 %)
  export   write every finished run to runs/unified/<fire>_<engine>.npz
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from fire_sim import annaburroo
from fire_sim import csiro_fds_added as added
from fire_sim import csiro_fds_full as full
from fire_sim import elmfire_csiro as elm
from fire_sim import wrf_sfire_csiro as wrf
from fire_sim.elmfire_csiro import Conditions, CsiroFire, Fuel
from fire_sim.paths import RUNS_DIR

UNIFIED_DIR = RUNS_DIR / "unified"
TIMING_FRACTION = 0.8

ELMFIRE = "ELMFIRE"
WRF_SFIRE = "WRF-SFIRE"
FDS_FULL = "FDS full physics"

BURNED_MEANS = {
    ELMFIRE: "the fire line has crossed the cell (ELMFIRE's time-of-arrival raster)",
    WRF_SFIRE: "the level-set function is below zero (TIGN_G gives the time)",
    FDS_FULL: f"the air 0.25 m above the ground is hotter than {full.ARRIVAL_TEMPERATURE_C:.0f} C",
}


@dataclass(frozen=True)
class FireRun:
    """One engine's run of one fire, on the common grid."""

    engine: str
    fire: CsiroFire
    case: str
    x_m: np.ndarray
    y_m: np.ndarray
    arrival_s: np.ndarray

    @property
    def cell_m(self) -> float:
        return float(abs(self.x_m[1] - self.x_m[0]))

    @property
    def burned_means(self) -> str:
        return BURNED_MEANS[self.engine]

    def burned_at(self, time_s: float) -> np.ndarray:
        return self.arrival_s <= time_s

    def on_plot(self) -> np.ndarray:
        """Cells on the plot; engines may burn a little past its edge, the real fire cannot."""
        xx, yy = np.meshgrid(self.x_m, self.y_m)
        return (
            (xx >= 0)
            & (xx <= self.fire.plot_length_m)
            & (np.abs(yy) <= self.fire.plot_half_width_m)
        )

    def burnt_area_m2(self, time_s: float) -> float:
        return float(np.sum(self.burned_at(time_s) & self.on_plot()) * self.cell_m**2)

    def head_distance_m(self, times_s: np.ndarray, bearing_deg: float) -> np.ndarray:
        """How far the fire has burned from the lighting line's centre along a compass bearing.

        The simulation frame is the real plot turned half a turn, so the bearing turns with it.
        """
        xx, yy = np.meshgrid(self.x_m, self.y_m)
        raster_bearing = np.radians(bearing_deg + 180.0)
        along = (xx - self.fire.lighting_x_m) * np.sin(raster_bearing) + (
            yy - self.fire.lighting_centre_y_m
        ) * np.cos(raster_bearing)
        return np.array(
            [along[self.burned_at(t)].max() if self.burned_at(t).any() else np.nan for t in times_s]
        )


def _grid_north_last(fire: CsiroFire) -> tuple[np.ndarray, np.ndarray]:
    x, y_north_first = elm.grid(fire)
    return x, y_north_first[::-1]


def from_elmfire(fire: CsiroFire, fuel: Fuel, conditions: Conditions) -> FireRun | None:
    case = elm.case_name(fire, fuel, conditions)
    folder = elm.RUNS_ELMFIRE_DIR / case
    if not any((folder / "outputs").glob("time_of_arrival_*.bil")):
        return None
    x, y = _grid_north_last(fire)
    return FireRun(ELMFIRE, fire, case, x, y, elm.read_arrival_times(folder)[::-1])


def from_wrf_sfire(fire: CsiroFire, fuel: Fuel, conditions: Conditions) -> FireRun | None:
    folder = wrf.case_dir(fire, fuel, conditions)
    if wrf.output_file(folder) is None:
        return None
    x, y = _grid_north_last(fire)
    return FireRun(WRF_SFIRE, fire, folder.name, x, y, wrf.read_arrival_times(fire, folder)[::-1])


MESH_CORNER = re.compile(r"&MESH IJK=[\d,]+, XB=([-\d.]+),[-\d.]+,([-\d.]+),")


def fds_full_case(fire: CsiroFire, *, turns_with_station: bool = False) -> full.FullCase:
    """The 1 m full-physics run with the field wind and lighting of NIST's two fires, turning
    with the station's direction if asked; the six added fires always turn with it."""
    if fire not in elm.FIRES:
        return next(c for c in added.added_cases() if c.fire.name == fire.name)
    return next(
        c
        for c in full.RUN_CASES
        if c.fire is fire
        and c.grid is full.Grid.ONE_METRE
        and c.conditions is Conditions.FIELD
        and c.turns_with_station is turns_with_station
    )


def fds_domain_corner(fds_input: str) -> tuple[float, float]:
    """South-west corner (x, y) of the meshes in an FDS input."""
    corners = [(float(x), float(y)) for x, y in MESH_CORNER.findall(fds_input)]
    return min(x for x, _ in corners), min(y for _, y in corners)


def from_fds_full(fire: CsiroFire, *, turns_with_station: bool = False) -> FireRun | None:
    """FDS's 1 m cells are centred half a metre in from its domain corner; its arrival map is
    put on ELMFIRE's grid cell by cell, nearest first (the grids differ by at most half a cell,
    and by the plot's far side where an added fire's domain is bigger)."""
    case = fds_full_case(fire, turns_with_station=turns_with_station)
    folder = full.RUNS_FULL_DIR / case.name
    if not full._completed(folder):
        return None
    x0, y0 = fds_domain_corner((folder / f"{case.name}.fds").read_text())
    _, _, arrival = full.ground_arrival(folder)
    x_fds = x0 + 0.5 + np.arange(arrival.shape[1])
    y_fds = y0 + 0.5 + np.arange(arrival.shape[0])
    x, y = _grid_north_last(fire)
    nearest = RegularGridInterpolator(
        (y_fds, x_fds), arrival, method="nearest", bounds_error=False, fill_value=np.nan
    )
    xx, yy = np.meshgrid(x, y)
    on_grid = nearest(np.column_stack([yy.ravel(), xx.ravel()])).reshape(xx.shape)
    return FireRun(FDS_FULL, fire, case.name, x, y, on_grid)


def runs_for(fire: CsiroFire) -> list[FireRun]:
    """Every engine's patchwork run of C064 or F19 with the field wind and lighting."""
    found = [
        from_elmfire(fire, Fuel.PATCHWORK, Conditions.FIELD),
        from_wrf_sfire(fire, Fuel.PATCHWORK, Conditions.FIELD),
        from_fds_full(fire),
    ]
    return [run for run in found if run is not None]


def time_to_reach(times_s: np.ndarray, head_m: np.ndarray, target_m: float) -> float:
    """First time the head is at target_m, interpolated; NaN if it never gets there."""
    reached = np.flatnonzero(head_m >= target_m)
    if reached.size == 0:
        return float("nan")
    index = reached[0]
    if index == 0:
        return float(times_s[0])
    return float(np.interp(target_m, head_m[index - 1 : index + 1], times_s[index - 1 : index + 1]))


def real_time_to_fraction(fire_name: str, fraction: float = TIMING_FRACTION) -> tuple[float, float]:
    """Target distance (m) and the time the real head reached it, from CSIRO's burn table."""
    burn = annaburroo.burn_periods(fire_name)
    target_m = fraction * burn.distance_m.iloc[-1]
    times = np.r_[0, burn.time_s.to_numpy()]
    return target_m, time_to_reach(times, np.r_[0, burn.distance_m.to_numpy()], target_m)


def engine_time_to_fraction(run: FireRun, fraction: float = TIMING_FRACTION) -> float:
    target_m, _ = real_time_to_fraction(run.fire.name, fraction)
    times = np.arange(0, run.fire.end_time_s + 1)
    head = run.head_distance_m(times, elm.travel_bearing(run.fire.name))
    return time_to_reach(times, np.nan_to_num(head, nan=-1.0), target_m)


def export() -> None:
    """Write each finished run's common-grid arrival map and a short description."""
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    for fire in elm.FIRES:
        for run in runs_for(fire):
            stem = f"{fire.name}_{run.engine.replace(' ', '_').lower()}"
            np.savez_compressed(
                UNIFIED_DIR / f"{stem}.npz",
                x_m=run.x_m,
                y_m=run.y_m,
                arrival_s=run.arrival_s.astype("float32"),
            )
            (UNIFIED_DIR / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "engine": run.engine,
                        "fire": fire.name,
                        "case": run.case,
                        "cell_m": run.cell_m,
                        "arrival_s": "seconds after lighting; NaN where the fire never arrived",
                        "frame": "x along the wind; y increases to the north; cell centres",
                        "burned_means": run.burned_means,
                    },
                    indent=2,
                )
            )
            print(f"{UNIFIED_DIR / stem}.npz")


def summary() -> None:
    """The engines side by side for C064 and F19."""
    rows = []
    for fire in elm.FIRES:
        _, real_s = real_time_to_fraction(fire.name)
        for run in runs_for(fire):
            end = fire.end_time_s
            reached_s = engine_time_to_fraction(run)
            rows.append(
                {
                    "fire": fire.name,
                    "engine": run.engine,
                    "area_at_end_m2": round(run.burnt_area_m2(end)),
                    "time_to_80pct_s": round(reached_s) if np.isfinite(reached_s) else None,
                    "real_s": round(real_s),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))
    for engine, meaning in BURNED_MEANS.items():
        print(f"{engine}: a cell has burned when {meaning}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["summary", "export"])
    {"summary": summary, "export": export}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
