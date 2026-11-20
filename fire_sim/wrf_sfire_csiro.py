"""CSIRO grassland fires in WRF-SFIRE, compared with the measurements.

WRF-SFIRE (github.com/openwfm/WRF-SFIRE) couples the Rothermel fire-spread formula, on a map of
1 m fire cells, to the WRF weather model on 10 m cells. It is a 2D fire model: the fire moves as a
line across the map and heats the air above it, but nothing burns in 3D as in FDS full physics.

This module writes the inputs of WRF's idealised fire case (`ideal.exe`, then `wrf.exe`) for each
fire from the same field data as the ELMFIRE and FDS translators, and reads the arrival times back
onto ELMFIRE's grid so the same scoring code measures all of them. The grass, plot and lighting
are the same as in ELMFIRE; the wind is a simplification: WRF's idealised case takes one steady
wind profile, so the measured 2 m wind is averaged over the run and, for the "station" cases, the
station's direction is averaged too. The weather model then develops the wind near the ground
itself; `compare` prints the wind the fire actually saw next to the one that was asked for.

Build WRF-SFIRE first (`~/git/WRF-SFIRE`, with `main/ideal.exe` and `main/wrf.exe`) and run it
with the MPI in the micromamba environment `~/micromamba/envs/wrf-sfire`. Set WRF_SFIRE_DIR and
WRF_ENV_BIN_DIR if they are somewhere else.

Commands:
  setup    write the WRF-SFIRE inputs to runs/wrf_sfire_csiro/<case>/
  run      run ideal.exe and wrf.exe on every case that has not produced output yet
  compare  C064 and F19: head positions vs measured and ELMFIRE; prints speeds, saves a chart
  score    all eight fires with the station's wind direction, against CSIRO's burn table
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from fire_sim import annaburroo
from fire_sim import elmfire_csiro as elm
from fire_sim.annaburroo import head_speed
from fire_sim.elmfire_csiro import Conditions, CsiroFire, Fuel, Grass
from fire_sim.paths import CSIRO_RAW_DIR, RUNS_DIR

WRF_SFIRE_DIR = Path(os.environ.get("WRF_SFIRE_DIR", Path.home() / "git" / "WRF-SFIRE"))
WRF_ENV_BIN_DIR = Path(
    os.environ.get("WRF_ENV_BIN_DIR", Path.home() / "micromamba" / "envs" / "wrf-sfire" / "bin")
)
RUNS_WRF_DIR = RUNS_DIR / "wrf_sfire_csiro"

# One weather cell holds REFINEMENT x REFINEMENT fire cells, so the fire map has 1 m cells.
WEATHER_CELL_M = 10
REFINEMENT = int(WEATHER_CELL_M / elm.CELL_SIZE_M)
# WRF's lighting line starts at this second of the run; arrival times are counted from it.
IGNITION_START_S = 1.0
# How fast a lit patch grows to the lighting strip's radius: the 5 m strip catches within 0.5 s.
IGNITION_ROS_M_S = 5.0
HISTORY_INTERVAL_S = 10
START_TIME = datetime(2006, 2, 23, 12, 42, 0)

# The wind profile handed to WRF: logarithmic from the measured 2 m wind up to 50 m, then steady.
PROFILE_TOP_M = 50.0
SOUNDING_HEIGHTS_M = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0)
SURFACE_PRESSURE_MB = 1013.0
NEUTRAL_THETA_K = 300.0

# WRF-SFIRE's fuel table: the standard category of each CSIRO grass, and the table's own numbers
# for what the measured grass does not give (they are those of the Rothermel grass models).
NO_FUEL_STANDARD_CATEGORY = 14
WIND_REDUCTION = {1: 0.36, 3: 0.44}
FUEL_PARTICLE_DENSITY = 32.0
TOTAL_MINERAL_CONTENT = 0.0555
EFFECTIVE_MINERAL_CONTENT = 0.010
FUEL_TIME_LAG_WEIGHT = 7.0
FEET_PER_M = elm.FEET_PER_M
NO_FUEL_LOAD = 1e-7

FIELD_CASES = (
    (Fuel.STANDARD, Conditions.FIELD),
    (Fuel.MEASURED, Conditions.FIELD),
    (Fuel.PATCHWORK, Conditions.FIELD),
)


def runs_for(fire: CsiroFire) -> tuple[tuple[Fuel, Conditions], ...]:
    """The WRF-SFIRE runs of a fire: wind straight down the plot for C064 and F19, and the
    station's direction for all eight."""
    if fire.name in annaburroo.FIRE_NAMES:
        return (*FIELD_CASES, *elm.STATION_CASES)
    return elm.STATION_CASES


def domain_cells(fire: CsiroFire) -> tuple[int, int]:
    """Weather cells across x and y: enough 10 m cells to cover ELMFIRE's whole map."""
    x, y = elm.grid(fire)
    return math.ceil(len(x) / WEATHER_CELL_M), math.ceil(len(y) / WEATHER_CELL_M)


def to_wrf_xy(fire: CsiroFire, x_m: float, y_m: float) -> tuple[float, float]:
    """Plot coordinates (x along the plot, y to the left of travel) from WRF's lower left corner."""
    x, y = elm.grid(fire)
    return x_m - (x[0] - elm.CELL_SIZE_M / 2), y_m - (y[-1] - elm.CELL_SIZE_M / 2)


def mean_wind_2m(fire: CsiroFire, conditions: Conditions) -> float:
    """The two upwind corners' 2 m wind (m/s), averaged over the run."""
    measured = annaburroo.upwind_wind(fire.name, corners=elm.upwind_corners(fire, conditions))
    run = measured[(measured.time_s >= 0) & (measured.time_s <= fire.end_time_s)]
    return float(((run.right_m_s + run.left_m_s) / 2).mean())


def mean_direction(fire: CsiroFire, conditions: Conditions) -> float:
    """Compass direction the wind comes from, in the simulation frame (270 = down the plot)."""
    if conditions is not Conditions.FIELD_STATION:
        return 270.0
    angles = np.radians(elm.station_directions(fire))
    return math.degrees(math.atan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360.0


def sounding(fire: CsiroFire, conditions: Conditions) -> str:
    """WRF's `input_sounding`: neutral air, wind equal to the measured 2 m wind at 2 m."""
    speed, from_deg = mean_wind_2m(fire, conditions), math.radians(mean_direction(fire, conditions))
    at_2m = math.log(2.0 / elm.ROUGHNESS_LENGTH_M)
    lines = [f"{SURFACE_PRESSURE_MB:.2f} {NEUTRAL_THETA_K:.2f} 0.000"]
    for height in SOUNDING_HEIGHTS_M:
        scale = math.log(min(height, PROFILE_TOP_M) / elm.ROUGHNESS_LENGTH_M) / at_2m
        u, v = -speed * scale * math.sin(from_deg), -speed * scale * math.cos(from_deg)
        lines.append(f"{height:.2f} {NEUTRAL_THETA_K:.2f} 0.000 {u:.3f} {v:.3f}")
    return "\n".join(lines) + "\n"


def case_grasses(fire: CsiroFire, fuel: Fuel) -> list[Grass]:
    """The measured grasses a case defines as WRF fuel categories; none for the standard grass."""
    if fuel is Fuel.STANDARD:
        return []
    if fuel is Fuel.MEASURED:
        return [elm.plot_average_grass(fire)]
    return elm.sample_grasses(fire)


def category_map(fire: CsiroFire, fuel: Fuel, conditions: Conditions) -> np.ndarray:
    """Fuel category of every fire cell on ELMFIRE's grid (north first), in WRF-SFIRE's numbers."""
    if fuel is Fuel.STANDARD:
        fuels = elm.fuel_map(fire, fire.standard_fuel_model, conditions)
        return np.where(fuels == elm.NONBURNABLE_FUEL, NO_FUEL_STANDARD_CATEGORY, fuels)
    if fuel is Fuel.MEASURED:
        fuels = elm.fuel_map(fire, elm.CUSTOM_FUEL, conditions)
    else:
        fuels = elm.patchwork_fuel_map(fire, conditions)
    no_fuel = len(case_grasses(fire, fuel)) + 1
    return np.where(fuels == elm.NONBURNABLE_FUEL, no_fuel, fuels - elm.CUSTOM_FUEL + 1)


def fuel_file(fire: CsiroFire, fuel: Fuel, conditions: Conditions) -> str:
    """WRF-SFIRE's `input_fc`: 'ni nj', then one row per west-east cell of south-north values.

    WRF-SFIRE's fire map has one weather cell more than the domain in each direction, and it
    can be larger than ELMFIRE's map to the north and east; that margin is left bare.
    """
    cells_x, cells_y = domain_cells(fire)
    no_fuel = (
        NO_FUEL_STANDARD_CATEGORY if fuel is Fuel.STANDARD else len(case_grasses(fire, fuel)) + 1
    )
    full = np.full(((cells_x + 1) * REFINEMENT, (cells_y + 1) * REFINEMENT), no_fuel, dtype=int)
    categories = category_map(fire, fuel, conditions)
    rows, columns = categories.shape
    full[:columns, :rows] = categories[::-1, :].T
    lines = [f"{full.shape[0]} {full.shape[1]}", *(" ".join(map(str, row)) for row in full)]
    return "\n".join(lines) + "\n"


def _array(values: list[float | str], *, quote: bool = False) -> str:
    return ", ".join(f"'{v}'" if quote else f"{v:g}" for v in values)


def fire_namelist(fire: CsiroFire, fuel: Fuel) -> str:
    """WRF-SFIRE's `namelist.fire`: the dead-grass moisture and, for measured grass, the fuel table.

    Category n of a measured case is grass n; the category after the last grass is bare ground.
    The numbers the measurements do not give are those of the standard grass model of the plot.
    The roughness length and wind height stay at WRF-SFIRE's grass defaults: this build stops with
    an error when `fz0` or `fwh` are in the table.
    """
    moisture = fire.dead_moisture_percent / 100
    grasses = case_grasses(fire, fuel)
    if not grasses:
        return f"&fuel_scalars\n fuelmc_g = {moisture:g},\n/\n"
    count = len(grasses)
    rows = [*grasses, Grass(NO_FUEL_LOAD, 0.305, 3500 * FEET_PER_M)]
    names = [f"CSIRO grass {n}" for n in range(1, count + 1)] + ["no fuel"]
    depth = [g.height_m for g in rows]
    columns = {
        "windrf": [WIND_REDUCTION[fire.standard_fuel_model]] * count + [NO_FUEL_LOAD],
        "fgi": [g.load_kg_m2 for g in rows],
        "fueldepthm": depth,
        "savr": [g.surface_to_volume_per_m / FEET_PER_M for g in rows],
        "fuelmce": [elm.FINE_GRASS_EXTINCTION_PERCENT / 100] * (count + 1),
        "fueldens": [FUEL_PARTICLE_DENSITY] * (count + 1),
        "st": [TOTAL_MINERAL_CONTENT] * (count + 1),
        "se": [EFFECTIVE_MINERAL_CONTENT] * (count + 1),
        "weight": [FUEL_TIME_LAG_WEIGHT] * (count + 1),
    }
    table = "\n".join(f" {name} = {_array(values)}," for name, values in columns.items())
    return (
        f"&fuel_scalars\n fuelmc_g = {moisture:g},\n nfuelcats = {count},\n"
        f" no_fuel_cat = {count + 1},\n/\n\n"
        f"&fuel_categories\n fuel_name = {_array(names, quote=True)},\n{table}\n/\n"
    )


def ignition_lines(fire: CsiroFire) -> list[dict[str, float]]:
    """Two lines from the lighting line's centre to its ends, walked at the crews' pace."""
    across_x = math.sin(math.radians(fire.lighting_tilt_deg))
    across_y = math.cos(math.radians(fire.lighting_tilt_deg))
    half, pace = fire.field_lighting_half_length_m, fire.field_walking_speed_m_s
    centre = (fire.lighting_x_m, fire.lighting_centre_y_m)
    lines = []
    for sign in (1, -1):
        end = (centre[0] + sign * half * across_x, centre[1] + sign * half * across_y)
        (sx, sy), (ex, ey) = to_wrf_xy(fire, *centre), to_wrf_xy(fire, *end)
        lines.append(
            {
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "start_time": IGNITION_START_S,
                "end_time": IGNITION_START_S + half / pace,
            }
        )
    return lines


def input_namelist(fire: CsiroFire) -> str:
    """WRF's `namelist.input`: the idealised fire case with this fire's domain, run time and lines.

    The weather and fire settings are those of the mosaic test case shipped with WRF-SFIRE.
    """
    cells_x, cells_y = domain_cells(fire)
    run_s = math.ceil(fire.end_time_s + IGNITION_START_S)
    end = START_TIME + timedelta(seconds=run_s)
    ignitions = "\n".join(
        f" fire_ignition_ros{n} = {IGNITION_ROS_M_S:g},\n"
        f" fire_ignition_start_x{n} = {line['start_x']:.2f},\n"
        f" fire_ignition_start_y{n} = {line['start_y']:.2f},\n"
        f" fire_ignition_end_x{n} = {line['end_x']:.2f},\n"
        f" fire_ignition_end_y{n} = {line['end_y']:.2f},\n"
        f" fire_ignition_radius{n} = {elm.LIGHTING_STRIP_HALF_WIDTH_M:g},\n"
        f" fire_ignition_start_time{n} = {line['start_time']:g},\n"
        f" fire_ignition_end_time{n} = {line['end_time']:.2f},"
        for n, line in enumerate(ignition_lines(fire), start=1)
    )
    return f""" &time_control
 run_days = 0,
 run_hours = 0,
 run_minutes = 0,
 run_seconds = {run_s},
 start_year = {START_TIME.year}, 0001, 0001,
 start_month = {START_TIME.month:02d}, 01, 01,
 start_day = {START_TIME.day:02d}, 01, 01,
 start_hour = {START_TIME.hour:02d}, 00, 00,
 start_minute = {START_TIME.minute:02d}, 01, 01,
 start_second = 00, 00, 00,
 end_year = {end.year}, 0001, 0001,
 end_month = {end.month:02d}, 01, 01,
 end_day = {end.day:02d}, 01, 01,
 end_hour = {end.hour:02d}, 00, 00,
 end_minute = {end.minute:02d}, 600, 600,
 end_second = {end.second:02d}, 00, 00,
 history_interval_s = {HISTORY_INTERVAL_S}, 30, 30,
 frames_per_outfile = 1000, 1000, 1000,
 restart = .false.,
 restart_interval = 5,
 io_form_history = 2,
 io_form_restart = 2,
 io_form_input = 2,
 io_form_boundary = 2,
 debug_level = 1,
/
 &domains
 time_step = 0,
 time_step_fract_num = 1,
 time_step_fract_den = 50,
 max_dom = 1,
 s_we = 1, 1, 1,
 e_we = {cells_x + 1}, 43, 43,
 s_sn = 1, 1, 1,
 e_sn = {cells_y + 1}, 43, 43,
 s_vert = 1, 1, 1,
 e_vert = 41, 41, 41,
 dx = {WEATHER_CELL_M}, 30, 10,
 dy = {WEATHER_CELL_M}, 30, 10,
 ztop = 600, 1500, 1500,
 grid_id = 1, 2, 3,
 parent_id = 0, 1, 2,
 i_parent_start = 0, 1, 1,
 j_parent_start = 0, 1, 1,
 parent_grid_ratio = 1, 2, 3,
 parent_time_step_ratio = 1, 2, 3,
 feedback = 1,
 smooth_option = 0,
 sr_x = {REFINEMENT}, 0, 0,
 sr_y = {REFINEMENT}, 0, 0,
/
 &physics
 mp_physics = 0, 0, 0,
 ra_lw_physics = 0, 0, 0,
 ra_sw_physics = 0, 0, 0,
 radt = 30, 30, 30,
 sf_sfclay_physics = 1, 0, 0,
 sf_surface_physics = 1, 0, 0,
 bl_pbl_physics = 0, 0, 0,
 bldt = 0, 0, 0,
 cu_physics = 0, 0, 0,
 cudt = 0, 0, 0,
 isfflx = 1,
 ifsnow = 0,
 icloud = 0,
 num_soil_layers = 5,
 mp_zero_out = 0,
/
 &fdda
/
 &dynamics
 rk_ord = 3,
 diff_opt = 2,
 km_opt = 2,
 damp_opt = 2,
 zdamp = 100., 5000., 5000.,
 dampcoef = 0.2, 0.2, 0.2,
 khdif = 0.05, 0.05, 0.05,
 kvdif = 0.05, 0.05, 0.05,
 smdiv = 0.1, 0.1, 0.1,
 emdiv = 0.01, 0.01, 0.01,
 epssm = 0.1, 0.1, 0.1,
 mix_full_fields = .true., .true., .true.,
 non_hydrostatic = .true., .true., .true.,
 h_mom_adv_order = 5, 5, 5,
 v_mom_adv_order = 3, 3, 3,
 h_sca_adv_order = 5, 5, 5,
 v_sca_adv_order = 3, 3, 3,
 time_step_sound = 20, 20, 20,
 moist_adv_opt = 1, 1, 1,
 scalar_adv_opt = 1, 1, 1,
/
 &bdy_control
 periodic_x = .false., .false., .false.,
 symmetric_xs = .false., .false., .false.,
 symmetric_xe = .false., .false., .false.,
 open_xs = .true., .false., .false.,
 open_xe = .true., .false., .false.,
 periodic_y = .false., .false., .false.,
 symmetric_ys = .false., .false., .false.,
 symmetric_ye = .false., .false., .false.,
 open_ys = .true., .false., .false.,
 open_ye = .true., .false., .false.,
 nested = .false., .true., .true.,
/
 &grib2
/
 &namelist_quilt
 nio_tasks_per_group = 0,
 nio_groups = 1,
/
 &fire
 ifire = 2,
 fire_fuel_read = 2,
 fire_fuel_cat = 1,
 fire_num_ignitions = 2,
{ignitions}
 fire_print_msg = 0,
 fire_print_file = 0,
 fire_const_time = -1.,
 fire_const_grnhfx = -1,
 fire_const_grnqfx = -1,
 fire_test_steps = 0,
 fire_mountain_type = 0,
 fire_topo_from_atm = 1,
 delt_perturbation = 0.0,
 stretch_grd = .true.,
 stretch_hyp = .true.,
 z_grd_scale = 2.35,
 sfc_full_init = .true.,
 sfc_lu_index = 28,
 sfc_tsk = 285.0,
 sfc_tmn = 280.0,
 fire_read_atm_ht = .false.,
 fire_read_lu = .false.,
 fire_boundary_guard = -1,
 fire_fuel_left_irl = 2,
 fire_fuel_left_jrl = 2,
 fire_atm_feedback = 1.0,
 fire_back_weight = 0.5,
 fire_grows_only = 1,
 fire_viscosity = 0.4,
 fire_upwinding = 3,
 fire_fuel_left_method = 1,
 fire_lfn_ext_up = 1.0,
 fire_advection = 0,
/
"""


def case_dir(fire: CsiroFire, fuel: Fuel, conditions: Conditions) -> Path:
    return RUNS_WRF_DIR / elm.case_name(fire, fuel, conditions)


def write_case(fire: CsiroFire, fuel: Fuel, conditions: Conditions) -> Path:
    """Write one case's input files and link the WRF-SFIRE programs and tables next to them."""
    folder = case_dir(fire, fuel, conditions)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "namelist.input").write_text(input_namelist(fire))
    (folder / "namelist.fire").write_text(fire_namelist(fire, fuel))
    (folder / "input_fc").write_text(fuel_file(fire, fuel, conditions))
    (folder / "input_sounding").write_text(sounding(fire, conditions))
    tables = [*(WRF_SFIRE_DIR / "run").glob("*.TBL"), *(WRF_SFIRE_DIR / "run").glob("*_DATA")]
    programs = [WRF_SFIRE_DIR / "main" / f"{name}.exe" for name in ("ideal", "wrf")]
    for source in [*tables, *programs]:
        (folder / source.name).symlink_to(source)
    return folder


def output_file(folder: Path) -> Path | None:
    outputs = sorted(folder.glob("wrfout_d01_*"))
    return outputs[-1] if outputs else None


def run_case(folder: Path) -> None:
    """Run ideal.exe, then wrf.exe, on one process; each writes its own log."""
    env = {**os.environ, "PATH": f"{WRF_ENV_BIN_DIR}:{os.environ['PATH']}"}
    for program in ("ideal", "wrf"):
        with (folder / f"{program}.log").open("w") as log:
            subprocess.run(
                ["mpirun", "-np", "1", f"./{program}.exe"],
                cwd=folder,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
    if "SUCCESS COMPLETE WRF" not in (folder / "rsl.out.0000").read_text():
        raise RuntimeError(f"{folder.name}: WRF did not finish, see {folder / 'rsl.error.0000'}")


def arrival_on_elmfire_grid(
    fire: CsiroFire, ignition_time_s: np.ndarray, level_set: np.ndarray
) -> np.ndarray:
    """Seconds after lighting each cell burned on ELMFIRE's grid (north first); NaN if unburned.

    WRF's arrays run south to north in rows and west to east in columns, and its map is padded
    on the north and east; a cell counts as burned where the level-set function is negative.
    """
    x, y = elm.grid(fire)
    arrival = np.where(level_set < 0, ignition_time_s - IGNITION_START_S, np.nan)
    return arrival[: len(y), : len(x)][::-1, :]


def read_arrival_times(fire: CsiroFire, folder: Path) -> np.ndarray:
    path = output_file(folder)
    with netcdf_file(path, mmap=False) as nc:
        ignition_time = np.array(nc.variables["TIGN_G"][-1])
        level_set = np.array(nc.variables["LFN"][-1])
    return arrival_on_elmfire_grid(fire, ignition_time, level_set)


def wind_at_fire(fire: CsiroFire, folder: Path) -> float:
    """Mean wind speed (m/s) on the fire map over the plot, at the first output after lighting."""
    with netcdf_file(output_file(folder), mmap=False) as nc:
        u, v = np.array(nc.variables["UF"][1]), np.array(nc.variables["VF"][1])
    x, y = elm.grid(fire)
    on_plot = (
        (x[None, :] >= 0)
        & (x[None, :] <= fire.plot_length_m)
        & (np.abs(y[:, None]) <= fire.plot_half_width_m)
    )
    speed = np.hypot(u, v)[: len(y), : len(x)][::-1, :]
    return float(speed[on_plot].mean())


def finished_cases(
    fire: CsiroFire, cases: tuple[tuple[Fuel, Conditions], ...]
) -> list[tuple[Fuel, Conditions, Path]]:
    folders = [(fuel, conditions, case_dir(fire, fuel, conditions)) for fuel, conditions in cases]
    return [(f, c, folder) for f, c, folder in folders if output_file(folder) is not None]


def score() -> None:
    """Head distance of every fire's station-wind runs against CSIRO's burn table."""
    rows = []
    figure, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, fire in zip(axes.flat, elm.all_fires(), strict=True):
        measured = annaburroo.burn_periods(fire.name)
        times, real = measured.time_s.to_numpy(), measured.distance_m.to_numpy()
        bearing = elm.travel_bearing(fire.name)
        ax.plot(times, real, "ko", label="CSIRO")
        curve_times = np.arange(0, fire.end_time_s + 1)
        for (fuel, _conditions, folder), style in zip(
            finished_cases(fire, elm.STATION_CASES), ("C2:", "C2--", "C2-"), strict=False
        ):
            arrival = read_arrival_times(fire, folder)
            model = elm.head_distances(fire, arrival, times, bearing)
            rows.append(
                {
                    "fire": fire.name,
                    "grass": fuel.value,
                    "real_m_s": head_speed(times, real),
                    "wrf_sfire_m_s": head_speed(times, model),
                    "real_last_m": real[-1],
                    "wrf_sfire_last_m": model[-1],
                }
            )
            label = "patchwork" if fuel is Fuel.PATCHWORK else f"{fuel.value} grass"
            ax.plot(
                curve_times,
                elm.head_distances(fire, arrival, curve_times, bearing),
                style,
                label=f"WRF-SFIRE, {label}",
            )
        ax.set(
            title=f"{fire.name} (bearing {bearing:.0f}°)",
            xlabel="Time since lighting (s)",
            ylabel="Head distance (m)",
        )
        ax.legend(fontsize=7)
    figure.tight_layout()
    chart = RUNS_WRF_DIR / "eight_fires_heads.png"
    figure.savefig(chart, dpi=130)
    table = pd.DataFrame(rows)
    table["share_of_real"] = table.wrf_sfire_m_s / table.real_m_s
    table.to_csv(RUNS_WRF_DIR / "eight_fires_scores.csv", index=False)
    print(table.round(2).to_string(index=False))
    print(f"Chart: {chart}")


def compare() -> None:
    """C064 and F19: head speeds and positions over time, next to the measurements and ELMFIRE."""
    figure, axes = plt.subplots(1, len(elm.FIRES), figsize=(11, 4.6))
    for ax, fire in zip(axes, elm.FIRES, strict=True):
        measured = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_exp.csv", skiprows=1)
        times = measured.Time.to_numpy()
        print(f"{fire.name}: measured {head_speed(times, measured.Front.to_numpy()):.2f} m/s")
        ax.plot(times, measured.Front, "ko", label="Measured")
        curve_times = np.arange(0, fire.end_time_s + 1)
        for fuel, conditions, folder in finished_cases(fire, FIELD_CASES):
            arrival = read_arrival_times(fire, folder)
            speed = head_speed(times, elm.head_positions(fire, arrival, times))
            asked = mean_wind_2m(fire, conditions)
            print(
                f"  WRF-SFIRE {elm.case_label(fuel, conditions):45s} {speed:.2f} m/s"
                f"   (2 m wind asked {asked:.1f}, fire's wind {wind_at_fire(fire, folder):.1f} m/s)"
            )
            ax.plot(
                curve_times,
                elm.head_positions(fire, arrival, curve_times),
                label=f"WRF-SFIRE, {fuel.value} grass",
            )
        elmfire = elm.RUNS_ELMFIRE_DIR / elm.case_name(fire, Fuel.PATCHWORK, Conditions.FIELD)
        if any((elmfire / "outputs").glob("time_of_arrival_*.bil")):
            arrival = elm.read_arrival_times(elmfire)
            speed = head_speed(times, elm.head_positions(fire, arrival, times))
            print(f"  ELMFIRE   {'measured grass patchwork, measured wind':45s} {speed:.2f} m/s")
            ax.plot(
                curve_times, elm.head_positions(fire, arrival, curve_times), "C3--", label="ELMFIRE"
            )
        ax.set(
            title=f"CSIRO {fire.name}",
            xlabel="Time (s)",
            ylabel="Fire head x (m)",
            ylim=(0, fire.plot_length_m * 1.1),
        )
        ax.legend(fontsize=8)
    figure.tight_layout()
    chart = RUNS_WRF_DIR / "front_positions.png"
    figure.savefig(chart, dpi=150)
    print(f"Chart: {chart}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("setup", "run", "compare", "score"))
    command = parser.parse_args().command
    cases = [
        (fire, fuel, conditions) for fire in elm.all_fires() for fuel, conditions in runs_for(fire)
    ]
    if command == "setup":
        for fire, fuel, conditions in cases:
            print(f"Wrote {write_case(fire, fuel, conditions)}")
    elif command == "run":
        for fire, fuel, conditions in cases:
            folder = case_dir(fire, fuel, conditions)
            if output_file(folder) is not None:
                continue
            print(f"Running {folder.name}", flush=True)
            run_case(folder)
    elif command == "compare":
        compare()
    else:
        score()


if __name__ == "__main__":
    main()
