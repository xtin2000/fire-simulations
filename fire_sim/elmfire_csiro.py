"""CSIRO grassland fires C064 and F19 in ELMFIRE, compared with the measurements and FDS.

ELMFIRE (github.com/lautenberger/elmfire) must be built in ~/elmfire, and GDAL's command-line
tools must be on PATH or in the micromamba env ~/micromamba/envs/elmfire.

Commands:
  setup    write the ELMFIRE inputs to runs/elmfire_csiro/<case>/
  run      run ELMFIRE on every case
  compare  fire-head positions vs measured and FDS; prints speeds, saves a chart
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fire_sim import annaburroo
from fire_sim.annaburroo import head_speed
from fire_sim.paths import CSIRO_RAW_DIR, CSIRO_RUNS_DIR, RUNS_DIR

ELMFIRE_DIR = Path(os.environ.get("ELMFIRE_DIR", Path.home() / "elmfire"))
ELMFIRE_BIN = ELMFIRE_DIR / "build" / "linux" / "bin" / "elmfire"
GDAL_BIN_DIR = Path(os.environ.get("GDAL_BIN_DIR", Path.home() / "micromamba/envs/elmfire/bin"))
RUNS_ELMFIRE_DIR = RUNS_DIR / "elmfire_csiro"
MEASURED_WIND_STEP_S = 5.0

CELL_SIZE_M = 1.0
NODATA = -9999
# Any projected system works on flat ground; this is the one ELMFIRE's tutorials use.
SPATIAL_REFERENCE = "EPSG:32610"
NONBURNABLE_FUEL = 99
# Custom grass fuel models: 50 for the plot average, 50 + n for fuel sample n of the patchwork.
CUSTOM_FUEL = 50
# The measured grass keeps fuel model 1 (short grass) where nothing was measured, as in FDS.
FINE_GRASS_EXTINCTION_PERCENT = 12
# Roughness length of NIST's FDS inputs, used to carry the 2 m wind up to ELMFIRE's 20 ft.
ROUGHNESS_LENGTH_M = 0.03
TWENTY_FEET_M = 6.096
M_PER_S_TO_MPH = 1 / 0.44704
KG_PER_M2_TO_LB_PER_FT2 = 0.204816
FEET_PER_M = 1 / 0.3048
# NIST lights the centre 10 m at t = 0, then the line burns outward at 1 m/s (SURF 'LINE').
NIST_LIGHTING_CENTRE_HALF_M = 5.0
NIST_LIGHTING_SPEED_M_S = 1.0
LIGHTING_X_M = -2.5
# ELMFIRE reads up to 300 ignition points but tracks only 100 (ALREADY_IGNITED in
# elmfire_level_set.f90); beyond that the lighting goes wrong, so long lines get wider spacing.
MAX_IGNITIONS = 100
# NIST's head detector (DEVC 'x') looks at the strip -5 < y < 0.
HEAD_STRIP_Y_M = (-5.0, 0.0)


class Fuel(Enum):
    """Which grass description goes into ELMFIRE."""

    STANDARD = "standard"
    MEASURED = "measured"
    PATCHWORK = "patchwork"


class Conditions(Enum):
    """Wind and lighting: NIST's steady FDS setup, or as measured in the field.

    FIELD takes the upwind corners from the fuel sheet; FIELD_EAST takes the east corners, as
    CSIRO's burn table and the weather station imply. They differ only for C064.
    """

    NIST = "nist"
    FIELD = "field"
    FIELD_EAST = "field_east"

    @property
    def is_field(self) -> bool:
        return self is not Conditions.NIST


# The grass choice and conditions of each ELMFIRE run, per fire.
CASES = (
    (Fuel.STANDARD, Conditions.NIST),
    (Fuel.MEASURED, Conditions.NIST),
    (Fuel.MEASURED, Conditions.FIELD),
    (Fuel.PATCHWORK, Conditions.FIELD),
)
EAST_CORNER_CASE = (Fuel.PATCHWORK, Conditions.FIELD_EAST)


@dataclass(frozen=True)
class CsiroFire:
    """One CSIRO experiment as NIST set it up for FDS, plus the measured grass and lighting."""

    name: str
    plot_length_m: float
    plot_half_width_m: float
    lighting_half_length_m: float
    domain_margin_m: float
    wind_2m_m_s: float
    dead_moisture_percent: float
    standard_fuel_model: int
    grass_height_m: float
    grass_load_kg_m2: float
    grass_surface_to_volume_per_m: float
    end_time_s: float
    field_lighting_half_length_m: float
    field_walking_speed_m_s: float


# Field lighting (Mell et al. 2007): two workers walk outward from the centre, C064 25 m in
# 26 s, F19 88 m in 56 s.
FIRES = (
    CsiroFire("C064", 100, 50, 25, 10, 4.6, 6.3, 1, 0.21, 0.283, 9770, 120, 25, 25 / 26),
    CsiroFire("F19", 200, 100, 90, 20, 4.9, 5.8, 3, 0.51, 0.313, 12240, 180, 87.5, 88 / 56),
)


def wind_at_20ft_mph(wind_2m_m_s: float) -> float:
    """Carry a 2 m wind up to 20 ft with a log profile; ELMFIRE wants 20 ft wind in mph."""
    ratio = math.log(TWENTY_FEET_M / ROUGHNESS_LENGTH_M) / math.log(2.0 / ROUGHNESS_LENGTH_M)
    return wind_2m_m_s * ratio * M_PER_S_TO_MPH


@dataclass(frozen=True)
class Grass:
    """One measured grass description, in SI units."""

    load_kg_m2: float
    height_m: float
    surface_to_volume_per_m: float


def custom_fuel_row(fuel_id: int, grass: Grass) -> str:
    """A measured grass as a row of ELMFIRE's fuel table (Rothermel's US units)."""
    load = grass.load_kg_m2 * KG_PER_M2_TO_LB_PER_FT2
    surface_to_volume = grass.surface_to_volume_per_m / FEET_PER_M
    depth = grass.height_m * FEET_PER_M
    return (
        f"{fuel_id},CSIRO,.FALSE.,{load:.5f},0,0,0,0,{surface_to_volume:.0f},9999,9999,"
        f"{depth:.3f},{FINE_GRASS_EXTINCTION_PERCENT},8000"
    )


def plot_average_grass(fire: CsiroFire) -> Grass:
    return Grass(fire.grass_load_kg_m2, fire.grass_height_m, fire.grass_surface_to_volume_per_m)


def sample_grasses(fire: CsiroFire) -> list[Grass]:
    """The 16 field fuel samples, in the order of fuel_samples()."""
    samples = annaburroo.fuel_samples(fire.name)
    return [
        Grass(s.load_kg_m2, s.height_m, s.surface_to_volume_per_m) for s in samples.itertuples()
    ]


def cases_for(fire: CsiroFire) -> tuple[tuple[Fuel, Conditions], ...]:
    """CASES, plus the east-corner wind where it differs from the fuel sheet's corners."""
    if annaburroo.UPWIND_CORNERS[fire.name] == annaburroo.EAST_CORNERS:
        return CASES
    return (*CASES, EAST_CORNER_CASE)


def upwind_corners(fire: CsiroFire, conditions: Conditions) -> annaburroo.UpwindCorners:
    if conditions is Conditions.FIELD_EAST:
        return annaburroo.EAST_CORNERS
    return annaburroo.UPWIND_CORNERS[fire.name]


def lighting_half_length(fire: CsiroFire, conditions: Conditions) -> float:
    if conditions.is_field:
        return fire.field_lighting_half_length_m
    return fire.lighting_half_length_m


def lighting_points(
    fire: CsiroFire, conditions: Conditions = Conditions.NIST
) -> list[tuple[float, float, float]]:
    """(x, y, time) points along the lighting line, as close together as ELMFIRE allows."""
    if conditions.is_field:
        centre_half, speed = 0.0, fire.field_walking_speed_m_s
    else:
        centre_half, speed = NIST_LIGHTING_CENTRE_HALF_M, NIST_LIGHTING_SPEED_M_S
    half = int(lighting_half_length(fire, conditions))
    spacing = math.ceil((2 * half + 1) / MAX_IGNITIONS)
    # Spaced out from the centre so a point burns at t = 0; ELMFIRE stops at once if none does.
    offsets = range(spacing, half + 1, spacing)
    points = []
    for y in [*(-y for y in reversed(offsets)), 0, *offsets]:
        delay = max(abs(y) - centre_half, 0.0) / speed
        points.append((LIGHTING_X_M, float(y), delay))
    return points


def weather_times(fire: CsiroFire, conditions: Conditions) -> np.ndarray:
    """Start time of each weather band: one steady band, or one every 5 s."""
    if conditions is Conditions.NIST:
        return np.array([0.0])
    return np.arange(0.0, fire.end_time_s + MEASURED_WIND_STEP_S, MEASURED_WIND_STEP_S)


def _grid(fire: CsiroFire) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre x (columns) and y (rows, north first) of the domain."""
    margin = fire.domain_margin_m
    x = np.arange(-margin, fire.plot_length_m + margin, CELL_SIZE_M) + CELL_SIZE_M / 2
    y_top = fire.plot_half_width_m + margin
    y = y_top - CELL_SIZE_M / 2 - np.arange(int(2 * y_top / CELL_SIZE_M)) * CELL_SIZE_M
    return x, y


def wind_2m_maps(fire: CsiroFire, conditions: Conditions) -> list[np.ndarray]:
    """2 m wind map (m/s) for each weather band.

    NIST: one steady value everywhere. Field: the two upwind corners every 5 s, varying in a
    straight line across the plot between them and held beyond the plot's sides.
    """
    x, y = _grid(fire)
    if conditions is Conditions.NIST:
        return [np.full((len(y), len(x)), fire.wind_2m_m_s, dtype=np.float32)]
    measured = annaburroo.upwind_wind(fire.name, corners=upwind_corners(fire, conditions))
    left_share = np.clip((y + fire.plot_half_width_m) / (2 * fire.plot_half_width_m), 0, 1)
    maps = []
    for t in weather_times(fire, conditions):
        right = np.interp(t, measured.time_s, measured.right_m_s)
        left = np.interp(t, measured.time_s, measured.left_m_s)
        across = right + (left - right) * left_share
        maps.append(np.repeat(across[:, None], len(x), axis=1).astype(np.float32))
    return maps


def _burnable(fire: CsiroFire, conditions: Conditions) -> np.ndarray:
    """The plot plus the 5 m lighting strip."""
    x, y = _grid(fire)
    xx, yy = np.meshgrid(x, y)
    on_plot = (xx >= 0) & (xx <= fire.plot_length_m) & (np.abs(yy) <= fire.plot_half_width_m)
    strip_half = lighting_half_length(fire, conditions)
    on_strip = (xx >= -5) & (xx < 0) & (np.abs(yy) <= strip_half)
    return on_plot | on_strip


def fuel_map(
    fire: CsiroFire, fuel_model: int, conditions: Conditions = Conditions.NIST
) -> np.ndarray:
    """One grass on the plot and on the 5 m lighting strip, bare ground elsewhere."""
    burnable = _burnable(fire, conditions)
    return np.where(burnable, fuel_model, NONBURNABLE_FUEL).astype(np.int16)


def patchwork_fuel_map(fire: CsiroFire, conditions: Conditions) -> np.ndarray:
    """Each grass cell takes the fuel model of its nearest field sample."""
    x, y = _grid(fire)
    xx, yy = np.meshgrid(x, y)
    samples = annaburroo.fuel_samples(fire.name)
    distances = np.hypot(
        xx[..., None] - samples.x_m.to_numpy(), yy[..., None] - samples.y_m.to_numpy()
    )
    nearest = CUSTOM_FUEL + distances.argmin(axis=-1)
    return np.where(_burnable(fire, conditions), nearest, NONBURNABLE_FUEL).astype(np.int16)


def _gdal_env() -> dict[str, str]:
    return {**os.environ, "PATH": f"{GDAL_BIN_DIR}:{os.environ['PATH']}"}


def _write_geotiff(bands: list[np.ndarray], fire: CsiroFire, path: Path) -> None:
    """Write bands as a raw BIL raster with a header, then let GDAL turn it into a GeoTIFF."""
    x, y = _grid(fire)
    stack = np.stack(bands, axis=1)  # BIL order: row, band, column
    is_float = stack.dtype.kind == "f"
    raw = path.with_suffix(".bil")
    stack.astype("<f4" if is_float else "<i2").tofile(raw)
    raw.with_suffix(".hdr").write_text(
        f"BYTEORDER I\nLAYOUT BIL\nNROWS {len(y)}\nNCOLS {len(x)}\nNBANDS {len(bands)}\n"
        f"NBITS {32 if is_float else 16}\nPIXELTYPE {'FLOAT' if is_float else 'SIGNEDINT'}\n"
        f"ULXMAP {x[0]}\nULYMAP {y[0]}\nXDIM {CELL_SIZE_M}\nYDIM {CELL_SIZE_M}\nNODATA {NODATA}\n"
    )
    subprocess.run(
        [
            "gdal_translate",
            "-q",
            "-of",
            "GTiff",
            "-a_srs",
            SPATIAL_REFERENCE,
            "-a_nodata",
            str(NODATA),
            "-co",
            "COMPRESS=DEFLATE",
            str(raw),
            str(path),
        ],
        check=True,
        env=_gdal_env(),
    )
    raw.unlink()
    raw.with_suffix(".hdr").unlink()


def _namelist(fire: CsiroFire, conditions: Conditions) -> str:
    ignitions = lighting_points(fire, conditions)
    band_step_s = fire.end_time_s if conditions is Conditions.NIST else MEASURED_WIND_STEP_S
    ignition_lines = "\n".join(
        f"X_IGN({i}) = {x}, Y_IGN({i}) = {y}, T_IGN({i}) = {t}"
        for i, (x, y, t) in enumerate(ignitions, start=1)
    )
    return f"""&INPUTS
FUELS_AND_TOPOGRAPHY_DIRECTORY = './inputs'
LANDSCAPE_FILENAME = 'landscape'
ADJ_FILENAME = 'adj'
PHI_FILENAME = 'phi'
DT_METEOROLOGY = {band_step_s}
WEATHER_DIRECTORY = './inputs'
WS_FILENAME = 'ws'
WD_FILENAME = 'wd'
M1_FILENAME = 'm1'
M10_FILENAME = 'm10'
M100_FILENAME = 'm100'
LH_MOISTURE_CONTENT = 30.0
LW_MOISTURE_CONTENT = 60.0
/
&OUTPUTS
OUTPUTS_DIRECTORY = './outputs'
DTDUMP = {fire.end_time_s}
DUMP_TIME_OF_ARRIVAL = .TRUE.
DUMP_SPREAD_RATE = .TRUE.
CONVERT_TO_GEOTIFF = .FALSE.
/
&TIME_CONTROL
SIMULATION_DT = 1.0
SIMULATION_DTMAX = 1.0
SIMULATION_TSTOP = {fire.end_time_s}
DT_INTERPOLATE_WIND = 1.0
/
&SIMULATOR
NUM_IGNITIONS = {len(ignitions)}
{ignition_lines}
WSMFEFF_LOW_MULT = 0.011364
/
&MISCELLANEOUS
PATH_TO_GDAL = 'auto'
SCRATCH = './scratch'
MISCELLANEOUS_INPUTS_DIRECTORY = './inputs'
FUEL_MODEL_FILE = 'fuel_models.csv'
/
"""


def case_name(fire: CsiroFire, fuel: Fuel, conditions: Conditions = Conditions.NIST) -> str:
    suffix = "" if conditions is Conditions.NIST else f"_{conditions.value}"
    return f"{fire.name}_{fuel.value}{suffix}"


def case_label(fuel: Fuel, conditions: Conditions) -> str:
    grass = "measured grass patchwork" if fuel is Fuel.PATCHWORK else f"{fuel.value} grass"
    if conditions is Conditions.FIELD_EAST:
        return f"{grass}, measured wind (east corners) and lighting"
    if conditions is Conditions.FIELD:
        return f"{grass}, measured wind and lighting"
    return grass


def write_case(fire: CsiroFire, fuel: Fuel, conditions: Conditions = Conditions.NIST) -> Path:
    """Write the rasters, fuel table and input file for one fire, grass and conditions."""
    case_dir = RUNS_ELMFIRE_DIR / case_name(fire, fuel, conditions)
    shutil.rmtree(case_dir, ignore_errors=True)
    inputs = case_dir / "inputs"
    for folder in (inputs, case_dir / "outputs", case_dir / "scratch"):
        folder.mkdir(parents=True)

    if fuel is Fuel.STANDARD:
        fuels, custom_rows = fuel_map(fire, fire.standard_fuel_model, conditions), []
    elif fuel is Fuel.MEASURED:
        fuels = fuel_map(fire, CUSTOM_FUEL, conditions)
        custom_rows = [custom_fuel_row(CUSTOM_FUEL, plot_average_grass(fire))]
    else:
        fuels = patchwork_fuel_map(fire, conditions)
        custom_rows = [
            custom_fuel_row(CUSTOM_FUEL + n, grass) for n, grass in enumerate(sample_grasses(fire))
        ]
    zeros = np.zeros_like(fuels)
    # Landscape band order: elevation, slope, aspect, fuel, canopy cover/height/base/density.
    _write_geotiff(
        [zeros, zeros, zeros, fuels, zeros, zeros, zeros, zeros], fire, inputs / "landscape.tif"
    )
    for name in ("adj", "phi"):
        _write_geotiff([np.ones(fuels.shape, dtype=np.float32)], fire, inputs / f"{name}.tif")
    wind_maps = wind_2m_maps(fire, conditions)
    _write_geotiff([wind_at_20ft_mph(u) for u in wind_maps], fire, inputs / "ws.tif")
    # The corner anemometers record speed only, and the site weather station's direction is not
    # used yet; the plots were lit at right angles to the prevailing wind, so it blows straight
    # down the plot (from the west in NIST's input).
    steady = {
        "wd": 270.0,
        "m1": fire.dead_moisture_percent,
        "m10": fire.dead_moisture_percent,
        "m100": fire.dead_moisture_percent,
    }
    for name, value in steady.items():
        bands = [np.full(fuels.shape, value, dtype=np.float32) for _ in wind_maps]
        _write_geotiff(bands, fire, inputs / f"{name}.tif")

    fuel_table = (ELMFIRE_DIR / "build" / "source" / "fuel_models.csv").read_text().rstrip("\n")
    (inputs / "fuel_models.csv").write_text("\n".join([fuel_table, *custom_rows]) + "\n")
    (inputs / "elmfire.data").write_text(_namelist(fire, conditions))
    return case_dir


def run_case(case_dir: Path) -> None:
    log = case_dir / "elmfire.log"
    with log.open("w") as log_file:
        subprocess.run(
            [str(ELMFIRE_BIN), "inputs/elmfire.data"],
            cwd=case_dir,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=_gdal_env(),
        )


def read_arrival_times(case_dir: Path) -> np.ndarray:
    """ELMFIRE's time-of-arrival raster in seconds; NaN where the fire never arrived."""
    raw = next((case_dir / "outputs").glob("time_of_arrival_*.bil"))
    header = dict(
        line.split()[:2]
        for line in raw.with_suffix(".hdr").read_text().splitlines()
        if line.strip()
    )
    rows, cols = int(header["NROWS"]), int(header["NCOLS"])
    dtype = "<f4" if header.get("BYTEORDER", "I") == "I" else ">f4"
    times = np.fromfile(raw, dtype=dtype).reshape(rows, cols)
    return np.where(times < 0, np.nan, times)


def head_positions(fire: CsiroFire, arrival_s: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Furthest downwind burned x in NIST's head strip, at each time."""
    x, y = _grid(fire)
    in_strip = (y > HEAD_STRIP_Y_M[0]) & (y < HEAD_STRIP_Y_M[1])
    strip = arrival_s[in_strip]
    positions = []
    for t in times_s:
        burned_columns = np.any(strip <= t, axis=0)
        positions.append(
            x[burned_columns].max() + CELL_SIZE_M / 2 if burned_columns.any() else np.nan
        )
    return np.array(positions)


def _fds_head(fire: CsiroFire) -> pd.DataFrame | None:
    devc = CSIRO_RUNS_DIR / f"Case_{fire.name}_LS" / f"Case_{fire.name}_LS_devc.csv"
    return pd.read_csv(devc, skiprows=1) if devc.exists() else None


def draw_arrival_maps() -> Path:
    """Map of when the fire reached each square, one panel per fire and grass choice."""
    figure, axes = plt.subplots(len(CASES), len(FIRES), figsize=(11, 5 * len(CASES)))
    for column, fire in enumerate(FIRES):
        measured = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_exp.csv", skiprows=1)
        outlines = pd.read_csv(
            CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_flame_position_data.csv"
        )
        x, y = _grid(fire)
        extent = (x[0] - 0.5, x[-1] + 0.5, y[-1] - 0.5, y[0] + 0.5)
        for row, (fuel, conditions) in enumerate(CASES):
            ax = axes[row, column]
            arrival = read_arrival_times(RUNS_ELMFIRE_DIR / case_name(fire, fuel, conditions))
            image = ax.imshow(
                arrival, extent=extent, cmap="inferno_r", vmin=0, vmax=fire.end_time_s
            )
            levels = np.arange(20, fire.end_time_s + 1, 20)
            contours = ax.contour(x, y, arrival, levels=levels, colors="white", linewidths=0.6)
            ax.clabel(contours, fmt="%d s", fontsize=7)
            half, length = fire.plot_half_width_m, fire.plot_length_m
            ax.plot([0, length, length, 0, 0], [-half, -half, half, half, -half], "c-", lw=1)
            ax.plot(measured.Front, np.zeros(len(measured)), "co", ms=6)
            for t in measured.Time:
                ax.plot(outlines[f"{t:.0f} x"], outlines[f"{t:.0f} y"], "c--", lw=1.2)
            for t, front in zip(measured.Time, measured.Front, strict=True):
                ax.annotate(
                    f"real {t:.0f} s", (front, 0), (front, 6), color="c", fontsize=7, ha="center"
                )
            ax.set(
                title=f"{fire.name}, {case_label(fuel, conditions)}",
                xlabel="x (m), wind →",
                ylabel="y (m)",
            )
            figure.colorbar(image, ax=ax, label="Fire arrival time (s)", shrink=0.8)
    figure.tight_layout()
    path = RUNS_ELMFIRE_DIR / "arrival_maps.png"
    figure.savefig(path, dpi=150)
    return path


def compare() -> None:
    """Print head speeds and draw head position over time for every fire."""
    figure, axes = plt.subplots(1, len(FIRES), figsize=(11, 4.6))
    for ax, fire in zip(axes, FIRES, strict=True):
        measured = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_exp.csv", skiprows=1)
        times = measured.Time.to_numpy()
        print(f"{fire.name}: measured {head_speed(times, measured.Front.to_numpy()):.2f} m/s")
        ax.plot(times, measured.Front, "ko", label="Measured")

        curve_times = np.arange(0, fire.end_time_s + 1)
        styles = ("C0-", "C0--", "C0:", "C0-.", "C3-")
        for (fuel, conditions), style in zip(cases_for(fire), styles, strict=False):
            case_dir = RUNS_ELMFIRE_DIR / case_name(fire, fuel, conditions)
            if not any((case_dir / "outputs").glob("time_of_arrival_*.bil")):
                continue
            arrival = read_arrival_times(case_dir)
            speed = head_speed(times, head_positions(fire, arrival, times))
            label = case_label(fuel, conditions)
            print(f"  ELMFIRE {label:42s} {speed:.2f} m/s")
            ax.plot(
                curve_times,
                head_positions(fire, arrival, curve_times),
                style,
                label=f"ELMFIRE, {label}",
            )

        fds = _fds_head(fire)
        if fds is not None:
            fds_speed = head_speed(times, np.interp(times, fds.Time, fds.x))
            print(f"  {'FDS level set (NIST run)':50s} {fds_speed:.2f} m/s")
            ax.plot(fds.Time, fds.x, "C1", label="FDS level set, 5 m")

        ax.set(
            title=f"CSIRO {fire.name}",
            xlabel="Time (s)",
            ylabel="Fire head x (m)",
            ylim=(0, fire.plot_length_m * 1.1),
        )
        ax.legend(fontsize=8)
    figure.tight_layout()
    chart = RUNS_ELMFIRE_DIR / "front_positions.png"
    figure.savefig(chart, dpi=150)
    print(f"Chart: {chart}")
    print(f"Maps: {draw_arrival_maps()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("setup", "run", "compare"))
    command = parser.parse_args().command
    cases = [(fire, fuel, conditions) for fire in FIRES for fuel, conditions in cases_for(fire)]
    if command == "setup":
        for fire, fuel, conditions in cases:
            print(f"Wrote {write_case(fire, fuel, conditions)}")
    elif command == "run":
        for fire, fuel, conditions in cases:
            case_dir = RUNS_ELMFIRE_DIR / case_name(fire, fuel, conditions)
            print(f"Running {case_dir.name}", flush=True)
            run_case(case_dir)
    else:
        compare()


if __name__ == "__main__":
    main()
