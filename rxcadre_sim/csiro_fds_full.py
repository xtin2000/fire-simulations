"""CSIRO fires C064 and F19 in FDS's full physics mode, with the field grass, wind and lighting.

NIST's validation files run each fire with grass blades as particles in an LES of the air (0.5 m
cells, and 1 m in the "crude" files), one plot-average grass and a steady wind. This module keeps
NIST's physics and grids and swaps in what the field crews measured, with nothing tuned to the
real fire:

  grass     the 16-sample patchwork (each patch its own load, height and grass type, which
            sets the blade thinness)
  wind      the mean of the two upwind corners every 5 s (the corners record speed only),
            blowing along the plot as in NIST's files or, in the "_dir" cases, turning with the
            site weather station's direction (10 m, one reading a minute)
  lighting  two walkers from the centre at the measured pace

For C064 the upwind corners are uncertain: the fuel sheet says SE and SW ("field"), CSIRO's burn
table and the weather station say NE and SE ("field_east"). Both are run.

F19 on the 0.5 m grid (9.2M cells) needs about 28 GB and swaps on a 24 GB Mac, so it only runs
on the 1 m grid here; C064 runs on both to show what the coarser grid changes.

Commands:
  download  fetch NIST's full-mode inputs and its published results
  setup     write runs/csiro_full/<case>/<case>.fds with one mesh group per MPI process
  run       run every unfinished case, one after another (writes run.log in each case)
  compare   fire-head positions against the real fire (NIST's photo positions and CSIRO's own
            burn table) and NIST's runs; prints speeds, saves a chart
  heads     fire-head speeds against CSIRO's burn table from the ground temperature slice (the
            method behind the headline results)
  maps      draw when the fire reached each point, with the real fire's outlines
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fdsreader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rxcadre_sim import annaburroo, elmfire_csiro
from rxcadre_sim.annaburroo import head_speed
from rxcadre_sim.csiro_fds_fast import _sample_extents, field_wind_lines, head_positions
from rxcadre_sim.elmfire_csiro import FIRES, Conditions, CsiroFire
from rxcadre_sim.paths import CSIRO_RAW_DIR, RUNS_DIR

NIST_FULL_URL = (
    "https://raw.githubusercontent.com/firemodels/fds/FDS-6.11.1/"
    "Validation/CSIRO_Grassland_Fires/FDS_Input_Files"
)
NIST_RESULTS_URL = "https://raw.githubusercontent.com/firemodels/out/master/CSIRO_Grassland_Fires"
FULL_INPUTS_DIR = CSIRO_RAW_DIR / "inputs_full"
NIST_RESULTS_DIR = CSIRO_RAW_DIR / "nist_full_results"
RUNS_FULL_DIR = RUNS_DIR / "csiro_full"
FDS_ENV = Path.home() / "FDS" / "FDS6" / "bin" / "fds_env.zsh"
# 36 meshes per fire; 12 ranks gave the best speed in the timing test (3 meshes each).
MPI_PROCESSES = 12
GROUND_SLICE_Z_M = 0.25
# Gas above 300 C just over the grass means flames, not the warm plume drifting ahead of them.
ARRIVAL_TEMPERATURE_C = 300.0
# The block maps are drawn north up and both fires ran west, along the simulation's +x.
PLOT_AXIS_BEARING_DEG = 270.0

MESH_AND_MULT = re.compile(r"&MESH IJK=([\d,]+), XB=([-\d.,]+), MULT_ID='mesh' /\n&MULT ([^/]*)/")
GRASS_SURF = re.compile(r"&SURF ID\s*=\s*'wet vegetation'.*?/", re.DOTALL)
GRASS_PART = re.compile(r"&PART ID='blade of grass'.*?/", re.DOTALL)
GRASS_INIT = re.compile(r"&INIT PART_ID='blade of grass'[^/]*/")
IGNITION_VENT = re.compile(r"&VENT XB=0\.0,([\d.]+),[^/]*SURF_ID='IGNITION'[^/]*/")
NIST_WIND = re.compile(r"&WIND [^/]*/")
NIST_OUTPUT_LINE = re.compile(r"^&(SLCF|BNDF)\b")


class Grid(Enum):
    """NIST's two grids, by the suffix of its file names."""

    HALF_METRE = ""
    ONE_METRE = "_crude"

    @property
    def label(self) -> str:
        return "0.5 m" if self is Grid.HALF_METRE else "1 m"


@dataclass(frozen=True)
class FullCase:
    """One full-mode run: fire, grid, upwind corners and whether the wind turns."""

    fire: CsiroFire
    grid: Grid = Grid.HALF_METRE
    conditions: Conditions = Conditions.FIELD
    turns_with_station: bool = False

    @property
    def name(self) -> str:
        turning = "_dir" if self.turns_with_station else ""
        grid = "" if self.grid is Grid.HALF_METRE else "_1m"
        return f"Case_{self.fire.name}_full_patchwork_{self.conditions.value}{turning}{grid}"

    @property
    def label(self) -> str:
        corners = elmfire_csiro.upwind_corners(self.fire, self.conditions)
        corners = f"{corners.right}+{corners.left}"
        turning = ", station direction" if self.turns_with_station else ""
        return f"field data ({corners} wind{turning}), {self.grid.label}"


C064, F19 = FIRES
RUN_CASES = (
    FullCase(C064),
    FullCase(C064, Grid.ONE_METRE),
    FullCase(F19, Grid.ONE_METRE),
    FullCase(C064, Grid.ONE_METRE, Conditions.FIELD_EAST),
    FullCase(C064, Grid.ONE_METRE, Conditions.FIELD_EAST, turns_with_station=True),
    FullCase(F19, Grid.ONE_METRE, turns_with_station=True),
)


def nist_name(fire: CsiroFire, grid: Grid) -> str:
    return f"Case_{fire.name}{grid.value}"


def download() -> None:
    """NIST's full-mode inputs, the grass chemistry they include, and NIST's head positions."""
    FULL_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    NIST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = [(f"{NIST_FULL_URL}/vegetation_model.txt", FULL_INPUTS_DIR / "vegetation_model.txt")]
    for fire in FIRES:
        for grid in Grid:
            name = nist_name(fire, grid)
            targets += [
                (f"{NIST_FULL_URL}/{name}.fds", FULL_INPUTS_DIR / f"{name}.fds"),
                (
                    f"{NIST_RESULTS_URL}/{name}_cat_devc.csv",
                    NIST_RESULTS_DIR / f"{name}_cat_devc.csv",
                ),
            ]
    for url, target in targets:
        subprocess.run(["curl", "-sfL", "-o", str(target), url], check=True)
    print(f"downloaded to {FULL_INPUTS_DIR} and {NIST_RESULTS_DIR}")


def explicit_meshes(text: str) -> str:
    """Expand NIST's MULT into MESH lines that name their MPI process.

    FDS only accepts fewer processes than meshes when each mesh says which process runs it.
    """
    match = MESH_AND_MULT.search(text)
    cells = match.group(1)
    x0, x1, y0, y1, z0, z1 = (float(v) for v in match.group(2).split(","))
    mult = dict(re.findall(r"(\w+)=([-\d.]+)", match.group(3)))
    dx, dy = float(mult["DX"]), float(mult["DY"])
    columns = range(int(mult["I_LOWER"]), int(mult["I_UPPER"]) + 1)
    rows = range(int(mult["J_LOWER"]), int(mult["J_UPPER"]) + 1)
    offsets = [(i, j) for j in rows for i in columns]
    per_process = -(-len(offsets) // MPI_PROCESSES)
    meshes = [
        f"&MESH IJK={cells}, XB={x0 + i * dx:g},{x1 + i * dx:g},{y0 + j * dy:g},{y1 + j * dy:g},"
        f"{z0:g},{z1:g}, MPI_PROCESS={n // per_process} /"
        for n, (i, j) in enumerate(offsets)
    ]
    return MESH_AND_MULT.sub("\n".join(meshes), text)


def _grass_type_lines(surf: str, part: str, grass: str) -> list[str]:
    """NIST's blade SURF and PART, renamed for one grass type with its blade thinness."""
    surface_to_volume = annaburroo.GRASS_SURFACE_TO_VOLUME_PER_M[grass]
    surf = surf.replace("'wet vegetation'", f"'wet vegetation {grass}'")
    surf = re.sub(
        r"SURFACE_VOLUME_RATIO\s*=\s*[\d.]+", f"SURFACE_VOLUME_RATIO = {surface_to_volume}.", surf
    )
    part = part.replace("'blade of grass'", f"'blade of grass {grass}'")
    part = part.replace("SURF_ID='wet vegetation'", f"SURF_ID='wet vegetation {grass}'")
    return [surf, part]


def patchwork_grass(text: str, fire: CsiroFire) -> str:
    """Replace NIST's single grass with one particle INIT per field sample."""
    samples = annaburroo.fuel_samples(fire.name)
    x_spans = _sample_extents(samples.x_m.to_numpy(), 0.0, fire.plot_length_m)
    half = fire.plot_half_width_m
    y_spans = _sample_extents(samples.y_m.to_numpy(), -half, half)
    surf, part = GRASS_SURF.search(text).group(0), GRASS_PART.search(text).group(0)
    types = []
    for grass in sorted(samples.grass.unique()):
        types += _grass_type_lines(surf, part, grass)
    inits = []
    for sample in samples.itertuples():
        (x0, x1), (y0, y1) = x_spans[sample.x_m], y_spans[sample.y_m]
        inits.append(
            f"&INIT PART_ID='blade of grass {sample.grass}', XB={x0:g},{x1:g},{y0:g},{y1:g},"
            f"0.00,{sample.height_m:g}, N_PARTICLES_PER_CELL=1, CELL_CENTERED=T, "
            f"MASS_PER_VOLUME={sample.load_kg_m2 / sample.height_m:.4f}, DRY=T /"
        )
    text = GRASS_SURF.sub("\n\n".join(types), text)
    text = GRASS_PART.sub("", text)
    return GRASS_INIT.sub("\n".join(inits), text)


def field_lighting(text: str, fire: CsiroFire) -> str:
    """NIST's ignitor strip, lit outward from the centre at the walkers' pace and reach."""
    width = IGNITION_VENT.search(text).group(1)
    half = fire.field_lighting_half_length_m
    return IGNITION_VENT.sub(
        f"&VENT XB=0.0,{width},{-half:g},{half:g},0.,0., SURF_ID='IGNITION', XYZ=0,0,0, "
        f"SPREAD_RATE={fire.field_walking_speed_m_s:.3f} /",
        text,
    )


def fds_direction(compass_from_deg: np.ndarray) -> np.ndarray:
    """Compass bearing the wind blows from, as FDS's direction in simulation axes.

    FDS also takes the direction the wind blows from, with 270 blowing along +x. A wind from
    behind the fires (the plot axis + 180) is therefore 270.
    """
    return (np.asarray(compass_from_deg) - PLOT_AXIS_BEARING_DEG + 90.0) % 360.0


def station_direction_lines(fire: CsiroFire) -> list[str]:
    """The weather station's wind direction as an FDS direction ramp."""
    station = annaburroo.weather_station_wind(fire.name)
    during = station[(station.time_s >= -60) & (station.time_s <= fire.end_time_s + 60)]
    return [
        f"&RAMP ID='station direction', T={t:g}, F={d:.1f} /"
        for t, d in zip(during.time_s, fds_direction(during.direction_deg), strict=True)
    ]


def field_wind(fire: CsiroFire, conditions: Conditions, *, turns_with_station: bool) -> str:
    lines = field_wind_lines(fire, conditions)
    if turns_with_station:
        lines[0] = lines[0].replace(" /", ", RAMP_DIRECTION_T='station direction' /")
        lines += station_direction_lines(fire)
    return "\n".join(lines)


def full_input(case: FullCase) -> str:
    """NIST's full-mode input for one fire with the field grass, wind and lighting."""
    fire = case.fire
    text = (FULL_INPUTS_DIR / f"{nist_name(fire, case.grid)}.fds").read_text()
    text = re.sub(r"CHID='[^']*'", f"CHID='{case.name}'", text)
    text = explicit_meshes(text)
    text = patchwork_grass(text, fire)
    text = field_lighting(text, fire)
    wind = field_wind(fire, case.conditions, turns_with_station=case.turns_with_station)
    text = NIST_WIND.sub(lambda _: wind, text)
    # NIST's slices come at three photo times; arrival maps need every second at the ground.
    text = re.sub(r"RAMP_SLCF\s*=\s*'s-ramp'\s*", "DT_SLCF=1. ", text)
    lines = [line for line in text.splitlines() if not NIST_OUTPUT_LINE.match(line)]
    tail = lines.index("&TAIL /")
    lines[tail:tail] = [
        f"&SLCF PBZ={GROUND_SLICE_Z_M}, QUANTITY='TEMPERATURE', CELL_CENTERED=T /",
        f"&SLCF PBZ={GROUND_SLICE_Z_M}, QUANTITY='HRRPUV', CELL_CENTERED=T /",
        "",
    ]
    return "\n".join(lines) + "\n"


def setup() -> None:
    for case in RUN_CASES:
        case_dir = RUNS_FULL_DIR / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"{case_dir.name}.fds").write_text(full_input(case))
        (case_dir / "vegetation_model.txt").write_text(
            (FULL_INPUTS_DIR / "vegetation_model.txt").read_text()
        )
        print(case_dir)


def run() -> None:
    """Run each unfinished case with MPI, in the order of RUN_CASES.

    F19's larger meshes overflow macOS's default 8 MB stack, so the stack is raised to the
    hard limit.
    """
    for case in RUN_CASES:
        case_dir = RUNS_FULL_DIR / case.name
        if _completed(case_dir):
            continue
        print(f"running {case_dir.name}", flush=True)
        command = (
            f"source {FDS_ENV} && export OMP_NUM_THREADS=1 && ulimit -s hard && "
            f"mpirun -n {MPI_PROCESSES} fds {case_dir.name}.fds > run.log 2>&1"
        )
        subprocess.run(["zsh", "-c", command], cwd=case_dir, check=True)


def _completed(case_dir: Path) -> bool:
    log = case_dir / "run.log"
    return log.exists() and "completed successfully" in log.read_text()


def furthest_head(x_m: np.ndarray, plot_length_m: float) -> np.ndarray:
    """Furthest point the fire has reached so far, on the plot.

    NIST's head detector follows the hottest gas, which jumps back to smouldering grass once the
    head burns out at the far edge and runs past the edge with the plume.
    """
    return np.minimum(np.maximum.accumulate(x_m), plot_length_m)


def _head_curves(fire: CsiroFire) -> list[tuple[str, str, pd.DataFrame]]:
    """(label, line style, Time/x table) for NIST's runs and ours, where they exist."""
    curves = []
    for grid, dash in zip(Grid, ("-", ":"), strict=True):
        nist = NIST_RESULTS_DIR / f"{nist_name(fire, grid)}_cat_devc.csv"
        if nist.exists():
            label = f"NIST inputs, {grid.label}"
            curves.append((label, "C7" + dash, pd.read_csv(nist, skiprows=1)))
    ours = [case for case in RUN_CASES if case.fire is fire]
    for colour, case in enumerate(ours, start=1):
        case_dir = RUNS_FULL_DIR / case.name
        if _completed(case_dir):
            dash = "-" if case.grid is Grid.HALF_METRE else ":"
            devices = pd.read_csv(case_dir / f"{case.name}_cat_devc.csv", skiprows=1)
            curves.append((case.label, f"C{colour}{dash}", devices))
    return curves


def compare() -> None:
    """Print head positions at the real fire's mapped times and draw them over time.

    Two records of the real head: NIST's (from Mell et al. 2007, the photo times) and CSIRO's
    burn table. Speeds are over NIST's times, as in the other engine comparisons.
    """
    figure, axes = plt.subplots(1, len(FIRES), figsize=(14, 6))
    for ax, fire in zip(axes, FIRES, strict=True):
        measured = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_exp.csv", skiprows=1)
        times = measured.Time.to_numpy()
        csiro = annaburroo.burn_periods(fire.name)
        print(f"{fire.name}: real fire {head_speed(times, measured.Front.to_numpy()):.2f} m/s")
        for source, at, heads in (
            ("NIST", times, measured.Front),
            ("CSIRO", csiro.time_s, csiro.distance_m),
        ):
            print(f"  real head ({source}) at {_list(at, 's')}: {_list(heads, 'm')}")
        for label, style, devices in _head_curves(fire):
            reached = furthest_head(devices.x.to_numpy(), fire.plot_length_m)
            speed = head_speed(times, np.interp(times, devices.Time, reached))
            at_nist = _list(np.interp(times, devices.Time, reached), "m")
            at_csiro = _list(np.interp(csiro.time_s, devices.Time, reached), "m")
            print(f"  {label:50s} {speed:.2f} m/s | NIST times {at_nist} | CSIRO times {at_csiro}")
            ax.plot(devices.Time, reached, style, label=f"{label} ({speed:.2f} m/s)")
        ax.plot(times, measured.Front, "ko", ms=7, label="Real fire (NIST)")
        ax.plot(csiro.time_s, csiro.distance_m, "ks", mfc="none", ms=8, label="Real fire (CSIRO)")
        ax.set(
            title=f"CSIRO {fire.name}, FDS full physics",
            xlabel="Time (s)",
            ylabel="Fire head x (m)",
            ylim=(0, fire.plot_length_m * 1.05),
        )
        ax.legend(fontsize=8)
    figure.tight_layout()
    chart = RUNS_FULL_DIR / "full_mode_compared.png"
    figure.savefig(chart, dpi=150)
    print(f"Chart: {chart}")


def first_hot_time(temperatures: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """First time each point passed ARRIVAL_TEMPERATURE_C (time on axis 0); NaN if it never did."""
    is_hot = temperatures > ARRIVAL_TEMPERATURE_C
    return np.where(is_hot.any(axis=0), times_s[is_hot.argmax(axis=0)], np.nan)


def ground_arrival(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y and the fire arrival time [y, x] from a run's ground temperature slice."""
    ground = fdsreader.Simulation(str(case_dir)).slices.filter_by_quantity("TEMPERATURE")[0]
    temperatures, coordinates = ground.to_global(masked=False, fill=np.nan, return_coordinates=True)
    arrival = first_hot_time(temperatures, np.asarray(ground.times))
    return coordinates["x"], coordinates["y"], arrival.T


def slice_heads(
    x: np.ndarray, y: np.ndarray, arrival: np.ndarray, times_s: np.ndarray, plot_length_m: float
) -> np.ndarray:
    """Furthest burned point on the plot inside NIST's head strip, at each time."""
    on_plot = np.where((x >= 0) & (x <= plot_length_m), arrival, np.nan)
    return head_positions(x, y, on_plot, times_s)


def _completed_cases(fire: CsiroFire) -> list[FullCase]:
    return [c for c in RUN_CASES if c.fire is fire and _completed(RUNS_FULL_DIR / c.name)]


def heads() -> None:
    """Head positions from the ground slice at CSIRO's mapped times, and speeds over them.

    compare uses NIST's head detector at NIST's photo times instead; the two differ by up to
    0.05 m/s. Use this one when comparing with CSIRO's burn table.
    """
    for fire in FIRES:
        csiro = annaburroo.burn_periods(fire.name)
        times = csiro.time_s.to_numpy()
        real_speed = head_speed(times, csiro.distance_m.to_numpy())
        print(
            f"{fire.name}: real head at {_list(times, 's')}: {_list(csiro.distance_m, 'm')}, "
            f"{real_speed:.2f} m/s"
        )
        for case in _completed_cases(fire):
            x, y, arrival = ground_arrival(RUNS_FULL_DIR / case.name)
            positions = slice_heads(x, y, arrival, times, fire.plot_length_m)
            speed = head_speed(times, positions)
            print(
                f"  {case.label:50s} {_list(positions, 'm')}, {speed:.2f} m/s "
                f"({speed / real_speed:.0%} of real)"
            )


def draw_maps() -> None:
    """Arrival map of every finished run, with NIST's real fire outlines at the photo times."""
    cases = [case for fire in FIRES for case in _completed_cases(fire)]
    figure, axes = plt.subplots(len(cases), 1, figsize=(9, 4.5 * len(cases)), squeeze=False)
    for ax, case in zip(axes[:, 0], cases, strict=True):
        fire = case.fire
        x, y, arrival = ground_arrival(RUNS_FULL_DIR / case.name)
        image = ax.pcolormesh(
            x, y, arrival, cmap="Blues", vmin=0, vmax=fire.end_time_s, shading="auto"
        )
        half, length = fire.plot_half_width_m, fire.plot_length_m
        ax.plot([0, length, length, 0, 0], [-half, -half, half, half, -half], "k-", lw=1)
        outlines = pd.read_csv(
            CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_flame_position_data.csv"
        )
        for column in outlines.columns[::2]:
            t = column.split()[0]
            ax.plot(outlines[f"{t} x"], outlines[f"{t} y"], "k--", lw=1.2)
            ax.annotate(f"real {t} s", (outlines[f"{t} x"].max(), 0), fontsize=7)
        ax.set(
            title=f"{fire.name}, {case.label}",
            xlabel="x (m), wind →",
            ylabel="y (m)",
            aspect="equal",
        )
        figure.colorbar(image, ax=ax, label="Fire arrival time (s)", shrink=0.8)
    figure.tight_layout()
    path = RUNS_FULL_DIR / "arrival_maps.png"
    figure.savefig(path, dpi=150)
    print(f"Maps: {path}")


def _list(values, unit: str) -> str:
    return "/".join(f"{v:.0f}" for v in values) + f" {unit}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = {
        "download": download,
        "setup": setup,
        "run": run,
        "compare": compare,
        "heads": heads,
        "maps": draw_maps,
    }
    parser.add_argument("command", choices=commands)
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
