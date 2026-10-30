"""RxCADRE 2012 S5 grass burn in FDS's fast level-set mode (mode 1), from the measured inputs.

Every input except the fuel-bed depth is read from the RxCADRE files (see s5_data). The depth is
the one tuned value: ``calibrate`` sweeps it and scores each run against the fire sensors.

Commands:
  setup      write runs/<chid>/<chid>.fds for the S5 base case (options override measured values)
  patchwork  write cases with a best-guess mix of grass, shrub and oak clumps, one per seed
  run        run every unfinished case under runs/ with FDS (scripts/run_fds_batch.sh)
  compare    fire arrival at the S5 sensors against the measurements: spread rate and timing error
  map        save each run's arrival-time grid (NPZ + CSV) and draw its arrival map
  calibrate  sweep the fuel-bed depth (1-minute wind, plus the steady wind), run and compare
"""

from __future__ import annotations

import argparse
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import fdsreader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as PolygonPath
from scipy.ndimage import gaussian_filter

from fire_sim import s5_data
from fire_sim.paths import PROJECT_ROOT, RAW_DIR, RUNS_DIR
from fire_sim.s5_data import ignition_centre_utm, load_perimeter_local, load_sensors_local

# Reported in RxCADRE field documentation (two igniters walking outward from the line centre).
IGNITER_WALK_SPEED_M_S = 1.34
# The one tuned input: makes the simulated spread match the S5 sensors (``calibrate``).
TUNED_FUEL_DEPTH_M = 0.42

DOMAIN_X_M = (-80.0, 140.0)
DOMAIN_Y_M = (-180.0, 60.0)
DOMAIN_TOP_M = 6.0

BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "run_fds_batch.sh"
DEFAULT_PARALLEL_JOBS = 8
NO_ARRIVAL = 1.0e5

TONS_ACRE_TO_KG_M2 = 0.22417
# Patch size is not measured; clumps a few metres across are a guess for grass, shrubs and oaks.
DEFAULT_PATCH_SIZE_M = 6.0

CALIBRATION_DEPTHS_CM = (30, 32, 34, 36, 38, 40, 41, 42, 43, 44, 46, 48, 50)
STEADY_CALIBRATION_DEPTHS_CM = (30, 33, 36, 37, 39, 42, 45)


# ── Building the FDS input ──


@dataclass
class S5Scenario:
    """Inputs for one S5-style level-set run. Use ``from_rxcadre`` to fill the measured values."""

    chid: str
    wind_speed_m_s: float
    wind_from_deg: float
    fuel_load_kg_m2: float
    dead_moisture: float
    ignition_half_length_m: float
    fuel_index: int = 1
    fuel_height_m: float = TUNED_FUEL_DEPTH_M
    wind_height_m: float | None = None
    cell_size_m: float = 2.0
    end_time_s: float = 2400.0
    ignition_xy_m: tuple[float, float] = (0.0, 0.0)
    extra_surf_params: dict[str, float] = field(default_factory=dict)
    # Minute-by-minute measured wind; empty means a steady wind at the mean speed and direction.
    wind_times_s: tuple[float, ...] = ()
    wind_speeds_m_s: tuple[float, ...] = ()
    wind_from_degs: tuple[float, ...] = ()

    @classmethod
    def from_rxcadre(cls, chid: str = "s5_base", **overrides: object) -> S5Scenario:
        """Measured S5 values read from the RxCADRE files (see s5_data), then any overrides."""
        speed, direction = s5_data.wind()
        _, _, half_length = s5_data.ignition_line()
        times, speeds, directions, _ = s5_data.wind_series()
        measured = {
            "wind_speed_m_s": speed.value,
            "wind_from_deg": direction.value,
            "wind_times_s": tuple(times),
            "wind_speeds_m_s": tuple(speeds),
            "wind_from_degs": tuple(directions),
            "fuel_load_kg_m2": s5_data.fuel_load().value,
            "dead_moisture": s5_data.dead_fuel_moisture().value,
            "ignition_half_length_m": half_length.value,
        }
        return cls(chid=chid, **{**measured, **overrides})


def _fuel_surf_line(surf_id: str, scenario: S5Scenario, *, extra: str = "") -> str:
    params = [
        f"ID='{surf_id}'",
        f"VEG_LSET_FUEL_INDEX={scenario.fuel_index}",
        f"VEG_LSET_SURF_LOAD={scenario.fuel_load_kg_m2:.4f}",
        f"VEG_LSET_HT={scenario.fuel_height_m:.3f}",
        f"VEG_LSET_M1={scenario.dead_moisture:.3f}",
    ]
    if scenario.wind_height_m is not None:
        params.append(f"VEG_LSET_WIND_HEIGHT={scenario.wind_height_m:.2f}")
    params += [f"{key}={value}" for key, value in scenario.extra_surf_params.items()]
    if extra:
        params.append(extra)
    return "&SURF " + ", ".join(params) + " /"


def plot_cell_mask(
    perimeter: np.ndarray, cell: float, ignition_box: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid edges and a [row, column] mask of cells inside the plot, ignition cells excluded.

    Cells under the ignition line are left out: FDS rejects a whole VENT that overlaps another.
    """
    polygon = PolygonPath(perimeter)
    x_edges = np.arange(DOMAIN_X_M[0], DOMAIN_X_M[1] + cell / 2, cell)
    y_edges = np.arange(DOMAIN_Y_M[0], DOMAIN_Y_M[1] + cell / 2, cell)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centres, y_centres)
    mask = polygon.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ign_x0, ign_x1, ign_y0, ign_y1 = ignition_box
    mask &= ~((xx > ign_x0) & (xx < ign_x1) & (yy > ign_y0) & (yy < ign_y1))
    return x_edges, y_edges, mask


def row_run_vents(
    x_edges: np.ndarray, y_edges: np.ndarray, labels: np.ndarray, surf_ids: list[str]
) -> list[str]:
    """One VENT per same-row run of equal labels; label -1 means no VENT (bare ground)."""
    vents = []
    for row, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:], strict=True)):
        run_start = 0
        row_labels = [*labels[row], -1]
        for index in range(1, len(row_labels)):
            if row_labels[index] == row_labels[run_start]:
                continue
            if row_labels[run_start] >= 0:
                vents.append(
                    f"&VENT XB={x_edges[run_start]:.1f},{x_edges[index]:.1f},"
                    f"{y0:.1f},{y1:.1f},0,0, "
                    f"SURF_ID='{surf_ids[row_labels[run_start]]}' /"
                )
            run_start = index
    return vents


def _plot_fuel_vents(
    perimeter: np.ndarray, cell: float, ignition_box: tuple[float, float, float, float]
) -> list[str]:
    """Stair-step the rotated plot polygon into one fuel VENT per same-row run of cells."""
    x_edges, y_edges, mask = plot_cell_mask(perimeter, cell, ignition_box)
    return row_run_vents(x_edges, y_edges, np.where(mask, 0, -1), ["S5 grass"])


def ignition_box(scenario: S5Scenario) -> tuple[float, float, float, float]:
    cell = scenario.cell_size_m
    centre_x, centre_y = scenario.ignition_xy_m
    half_cells = int(round(scenario.ignition_half_length_m / cell))
    return (
        centre_x - half_cells * cell,
        centre_x + half_cells * cell,
        centre_y,
        centre_y + cell,
    )


def _ignition_lines(scenario: S5Scenario) -> list[str]:
    """Drip-torch line lit from the centre outward by two igniters walking apart."""
    cell = scenario.cell_size_m
    centre_x, centre_y = scenario.ignition_xy_m
    half_cells = int(round(scenario.ignition_half_length_m / cell))
    lines = []
    for offset in range(-half_cells, half_cells):
        x0 = centre_x + offset * cell
        ignite_time = abs(x0 + cell / 2 - centre_x) / IGNITER_WALK_SPEED_M_S
        surf_id = f"IGN {offset:+d}"
        lines.append(
            _fuel_surf_line(
                surf_id, scenario, extra=f"VEG_LSET_IGNITE_TIME={ignite_time:.1f}, COLOR='RED'"
            )
        )
        lines.append(
            f"&VENT XB={x0:.1f},{x0 + cell:.1f},{centre_y:.1f},{centre_y + cell:.1f},0,0, "
            f"SURF_ID='{surf_id}' /"
        )
    return lines


def _sensor_devices(sensors: pd.DataFrame) -> list[str]:
    return [
        f"&DEVC ID='{row.Sensor_ID}', XYZ={row.x_local:.2f},{row.y_local:.2f},0, "
        "QUANTITY='FIRE ARRIVAL TIME', IOR=3 /"
        for row in sensors.itertuples()
    ]


def _wind_lines(scenario: S5Scenario) -> list[str]:
    """Steady wind, or the measured 1-minute series as FDS ramps (speed as a multiple of SPEED)."""
    if not scenario.wind_times_s:
        return [f"&WIND DIRECTION={scenario.wind_from_deg}, SPEED={scenario.wind_speed_m_s} /"]
    lines = [
        f"&WIND DIRECTION={scenario.wind_from_deg}, SPEED={scenario.wind_speed_m_s}, "
        "RAMP_SPEED_T='wind speed', RAMP_DIRECTION_T='wind dir' /"
    ]
    for time, speed in zip(scenario.wind_times_s, scenario.wind_speeds_m_s, strict=True):
        lines.append(
            f"&RAMP ID='wind speed', T={time:g}, F={speed / scenario.wind_speed_m_s:.3f} /"
        )
    for time, direction in zip(scenario.wind_times_s, scenario.wind_from_degs, strict=True):
        lines.append(f"&RAMP ID='wind dir', T={time:g}, F={direction:.1f} /")
    return lines


def build_input(scenario: S5Scenario, *, fuel_lines: list[str] | None = None) -> str:
    """Return the full FDS input text; ``fuel_lines`` replaces the one uniform grass fuel."""
    cell = scenario.cell_size_m
    nx = round((DOMAIN_X_M[1] - DOMAIN_X_M[0]) / cell)
    ny = round((DOMAIN_Y_M[1] - DOMAIN_Y_M[0]) / cell)
    nz = round(DOMAIN_TOP_M / cell)
    header = [
        f"&HEAD CHID='{scenario.chid}', TITLE='RxCADRE 2012 S5 level-set "
        f"(wind {scenario.wind_speed_m_s} m/s from {scenario.wind_from_deg} deg)' /",
        f"&MESH IJK={nx},{ny},{nz}, XB={DOMAIN_X_M[0]},{DOMAIN_X_M[1]},"
        f"{DOMAIN_Y_M[0]},{DOMAIN_Y_M[1]},0,{DOMAIN_TOP_M} /",
        f"&TIME T_END={scenario.end_time_s} /",
        "&MISC LEVEL_SET_MODE=1 /",
        "&RADI RADIATION=F /",
        *_wind_lines(scenario),
        "&DUMP DT_BNDF=10., DT_DEVC=10. /",
        *[
            f"&VENT MB='{side}', SURF_ID='OPEN' /"
            for side in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMAX")
        ],
        "&SURF ID='BARE', DEFAULT=T, COLOR='GRAY 60' /",
    ]
    if fuel_lines is None:
        fuel_lines = [
            _fuel_surf_line("S5 grass", scenario, extra="COLOR='GREEN'"),
            *_plot_fuel_vents(load_perimeter_local(), cell, ignition_box(scenario)),
        ]
    body = [
        *_ignition_lines(scenario),
        *fuel_lines,
        "&BNDF QUANTITY='FIRE ARRIVAL TIME' /",
        "&BNDF QUANTITY='LS SPREAD RATE' /",
        "&SLCF AGL_SLICE=0.1, QUANTITY='LEVEL SET VALUE' /",
        *_sensor_devices(load_sensors_local()),
        "&TAIL /",
    ]
    return "\n".join(header + body) + "\n"


def write_case(scenario: S5Scenario, out_dir: Path = RUNS_DIR) -> Path:
    case_dir = out_dir / scenario.chid
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / f"{scenario.chid}.fds"
    input_path.write_text(build_input(scenario))
    return input_path


# ── Best-guess fuel patchwork ──
# RxCADRE measured how much of each fuel S5 had (25 sample plots) but not where it grew, so the
# layout is invented: a smoothed random field, thresholded so each fuel covers its measured
# share. Several seeds show how much the (unknown) layout matters.


@dataclass(frozen=True)
class PatchFuel:
    """One fuel class of the patchwork and the FDS settings for it."""

    name: str
    fuel_index: int
    cover_fraction: float
    load_kg_m2: float
    height_m: float | None
    color: str
    live_woody_moisture: float | None = None


def _mean_row(path: str) -> pd.Series:
    table = pd.read_csv(RAW_DIR / path)
    return table[table.Category == "Mean"].iloc[0]


def _measured_moisture(category: str) -> float:
    table = pd.read_csv(RAW_DIR / "fuels/CADRE_2008_2011_2012_Fuel_moistures.csv")
    row = table[(table.Year == 2012) & (table.Site == "S5") & (table.Category == category)]
    return float(row.Mean_FM.iloc[0]) / 100


def patch_fuels(grass_height_m: float, *, separate_litter: bool = False) -> list[PatchFuel]:
    """Shrub/oak clumps plus grass, with cover and load matching the S5 means.

    Litter is spread under every patch. With ``separate_litter`` the ground between plants
    becomes litter-only patches; otherwise it is blended into the grass, because the measured
    bare soil lies between tufts centimetres apart, far below the 2 m grid.
    """
    mean = _mean_row("fuels/2012_S5.csv").apply(pd.to_numeric, errors="coerce")
    load = {
        key: mean[f"Pre_loading_{key}"] * TONS_ACRE_TO_KG_M2
        for key in ("grass", "forb", "shrub", "litter", "woody_1hr_10hr", "oak")
    }
    herb_cover = (mean.Pre_veg_cover_grass + mean.Pre_veg_cover_forb) / 100
    woody_cover = (mean.Pre_veg_cover_shrub + mean.Pre_veg_cover_oak) / 100
    ground_litter = load["litter"] + load["woody_1hr_10hr"]
    woody_height_cm = (
        mean.Pre_veg_cover_shrub * mean.Pre_avg_height_shrub
        + mean.Pre_veg_cover_oak * mean.Pre_avg_height_oak
    ) / (mean.Pre_veg_cover_shrub + mean.Pre_veg_cover_oak)
    woody = PatchFuel(
        "shrub and oak",
        fuel_index=5,
        cover_fraction=woody_cover,
        load_kg_m2=ground_litter + (load["shrub"] + load["oak"]) / woody_cover,
        height_m=woody_height_cm / 100,
        color="139,90,43",
        live_woody_moisture=_measured_moisture("FM_Shrub"),
    )
    grass_cover = herb_cover if separate_litter else 1 - woody_cover
    grass = PatchFuel(
        "grass",
        fuel_index=1,
        cover_fraction=grass_cover,
        load_kg_m2=ground_litter + (load["grass"] + load["forb"]) / grass_cover,
        height_m=grass_height_m,
        color="90,158,69",
    )
    if not separate_litter:
        return [woody, grass]
    litter = PatchFuel(
        "litter",
        fuel_index=9,
        cover_fraction=1 - herb_cover - woody_cover,
        load_kg_m2=ground_litter,
        height_m=None,
        color="217,201,138",
    )
    return [woody, grass, litter]


def patch_labels(
    mask: np.ndarray, fuels: list[PatchFuel], *, seed: int, patch_size_m: float, cell_m: float
) -> np.ndarray:
    """Label each plot cell with a fuel index (-1 outside the plot), clumped by a smooth field."""
    rng = np.random.default_rng(seed)
    smooth_field = gaussian_filter(rng.standard_normal(mask.shape), sigma=patch_size_m / cell_m / 2)
    ranks = smooth_field[mask].argsort()[::-1].argsort() / mask.sum()
    cumulative = np.cumsum([fuel.cover_fraction for fuel in fuels])
    labels = np.full(mask.shape, -1)
    labels[mask] = np.searchsorted(cumulative, ranks, side="right")
    return labels


def _patch_surf_line(fuel: PatchFuel, dead_moisture: float) -> str:
    params = [
        f"ID='{fuel.name}'",
        f"VEG_LSET_FUEL_INDEX={fuel.fuel_index}",
        f"VEG_LSET_SURF_LOAD={fuel.load_kg_m2:.4f}",
        f"VEG_LSET_M1={dead_moisture:.3f}",
    ]
    if fuel.height_m is not None:
        params.append(f"VEG_LSET_HT={fuel.height_m:.3f}")
    if fuel.live_woody_moisture is not None:
        params.append(f"VEG_LSET_MLW={fuel.live_woody_moisture:.3f}")
    params.append(f"RGB={fuel.color}")
    return "&SURF " + ", ".join(params) + " /"


def _plot_layout(path, x_edges, y_edges, labels, fuels) -> None:
    figure, axes = plt.subplots(figsize=(7, 7.5))
    colors = [tuple(int(v) / 255 for v in fuel.color.split(",")) for fuel in fuels]
    shown = np.ma.masked_less(labels, 0)
    axes.pcolormesh(
        x_edges, y_edges, shown, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(fuels) - 0.5
    )
    perimeter = load_perimeter_local()
    axes.plot(perimeter[:, 0], perimeter[:, 1], color="#2a9d8f", lw=1.5)
    axes.plot([-24.8, 24.8], [0, 0], color="#e63946", lw=3, label="ignition line")
    sensors = load_sensors_local()
    axes.scatter(sensors.x_local, sensors.y_local, c="#4cc9f0", edgecolor="black", s=40, zorder=5)
    for fuel, color in zip(fuels, colors, strict=True):
        share = (labels == fuels.index(fuel)).sum() / (labels >= 0).sum()
        axes.scatter([], [], marker="s", s=80, color=color, label=f"{fuel.name} ({share:.0%})")
    axes.set_aspect("equal")
    axes.set_xlabel("east of ignition centre (m)")
    axes.set_ylabel("north of ignition centre (m)")
    axes.set_title(f"{path.parent.name}: best-guess S5 fuel layout")
    axes.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def write_patchwork_case(
    chid: str, *, seed: int, patch_size_m: float, grass_height_m: float, separate_litter: bool
) -> None:
    """Write the FDS input and a fuel-layout picture for one patchwork layout."""
    scenario = S5Scenario.from_rxcadre(chid, fuel_height_m=grass_height_m)
    fuels = patch_fuels(grass_height_m, separate_litter=separate_litter)
    x_edges, y_edges, mask = plot_cell_mask(
        load_perimeter_local(), scenario.cell_size_m, ignition_box(scenario)
    )
    labels = patch_labels(
        mask, fuels, seed=seed, patch_size_m=patch_size_m, cell_m=scenario.cell_size_m
    )
    fuel_lines = [
        *[_patch_surf_line(fuel, scenario.dead_moisture) for fuel in fuels],
        *row_run_vents(x_edges, y_edges, labels, [fuel.name for fuel in fuels]),
    ]
    case_dir = RUNS_DIR / chid
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{chid}.fds").write_text(build_input(scenario, fuel_lines=fuel_lines))
    _plot_layout(case_dir / "fuel_layout.png", x_edges, y_edges, labels, fuels)


# ── Comparing with the fire sensors ──


@dataclass
class Comparison:
    """Model-versus-measurement summary for one run."""

    chid: str
    measured_ros_m_s: float
    model_ros_m_s: float
    rmse_s: float
    implied_ignition_utc: str
    per_sensor: pd.DataFrame


def spread_rate_from_arrivals(x: np.ndarray, y: np.ndarray, arrival_s: np.ndarray) -> float:
    """Head-fire rate of spread from a plane fit t = a + b x + c y (1 / |grad t|)."""
    design = np.c_[np.ones_like(x), x, y]
    coefficients, *_ = np.linalg.lstsq(design, arrival_s, rcond=None)
    return float(1.0 / np.hypot(coefficients[1], coefficients[2]))


def compare_run(run_dir: Path) -> Comparison:
    chid = run_dir.name
    devc = pd.read_csv(run_dir / f"{chid}_devc.csv", skiprows=1)
    final_arrivals = devc.iloc[-1].drop("Time")

    sensors = load_sensors_local().set_index("Sensor_ID")
    sensors["measured_s"] = pd.to_timedelta(sensors.arrival_utc).dt.total_seconds()
    sensors["model_s"] = final_arrivals.reindex(sensors.index).astype(float)
    sensors.loc[sensors.model_s >= NO_ARRIVAL, "model_s"] = np.nan

    reached = sensors.dropna(subset=["model_s"])
    offset = float((reached.measured_s - reached.model_s).mean())
    sensors["model_shifted_s"] = sensors.model_s + offset
    sensors["error_s"] = sensors.model_shifted_s - sensors.measured_s

    return Comparison(
        chid=chid,
        measured_ros_m_s=spread_rate_from_arrivals(
            sensors.x_local.values, sensors.y_local.values, sensors.measured_s.values
        ),
        model_ros_m_s=spread_rate_from_arrivals(
            reached.x_local.values, reached.y_local.values, reached.model_s.values
        ),
        rmse_s=float(np.sqrt(np.nanmean(sensors.error_s**2))),
        implied_ignition_utc=str(pd.to_timedelta(round(offset), unit="s")).split()[-1],
        per_sensor=sensors[["y_local", "measured_s", "model_s", "error_s"]],
    )


# ── Arrival-time grid and map ──


def arrival_time_grid(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, arrival[y, x]) with NaN where the fire never arrived."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level_set = fdsreader.Simulation(str(run_dir)).slices.filter_by_quantity("LEVEL SET VALUE")[
            0
        ]
        frames = level_set[0].data
    times = np.asarray(level_set.times)
    is_burned = frames > 0
    has_burned = is_burned.any(axis=0)
    arrival = np.where(has_burned, times[is_burned.argmax(axis=0)], np.nan).T
    x = np.linspace(*DOMAIN_X_M, arrival.shape[1])
    y = np.linspace(*DOMAIN_Y_M, arrival.shape[0])
    return x, y, arrival


def save_grid(run_dir: Path, x: np.ndarray, y: np.ndarray, arrival: np.ndarray) -> None:
    np.savez_compressed(
        run_dir / "arrival_time.npz",
        x_m=x,
        y_m=y,
        arrival_s=arrival,
        origin_utm16n=np.array(ignition_centre_utm()),
    )
    xx, yy = np.meshgrid(x, y)
    pd.DataFrame(
        {
            "x_m": xx.ravel(),
            "y_m": yy.ravel(),
            "easting": xx.ravel() + ignition_centre_utm()[0],
            "northing": yy.ravel() + ignition_centre_utm()[1],
            "arrival_s": arrival.ravel(),
        }
    ).dropna().to_csv(run_dir / "arrival_time.csv", index=False)


def plot_map(run_dir: Path, x: np.ndarray, y: np.ndarray, arrival: np.ndarray) -> Path:
    figure, axes = plt.subplots(figsize=(7, 7.5))
    minutes = arrival / 60
    mesh = axes.pcolormesh(x, y, minutes, cmap="inferno_r", shading="auto")
    contour_levels = np.arange(2, np.nanmax(minutes), 2)
    contours = axes.contour(x, y, minutes, levels=contour_levels, colors="white", linewidths=0.6)
    axes.clabel(contours, fmt="%d min", fontsize=7)

    perimeter = load_perimeter_local()
    axes.plot(perimeter[:, 0], perimeter[:, 1], color="#2a9d8f", lw=1.5, label="S5 burn block")
    axes.plot([-24.8, 24.8], [0, 0], color="#e63946", lw=3, label="ignition line")

    sensors = load_sensors_local()
    measured = pd.to_timedelta(sensors.arrival_utc).dt.total_seconds()
    axes.scatter(
        sensors.x_local,
        sensors.y_local,
        c="#4cc9f0",
        edgecolor="black",
        s=40,
        zorder=5,
        label="fire sensors (FB)",
    )
    for row, measured_s in zip(sensors.itertuples(), measured - measured.min(), strict=True):
        axes.annotate(
            f"{row.Sensor_ID}\n+{measured_s:.0f}s",
            (row.x_local, row.y_local),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=7,
            color="#4cc9f0",
        )

    axes.set_aspect("equal")
    axes.set_xlabel("east of ignition centre (m)")
    axes.set_ylabel("north of ignition centre (m)")
    axes.set_title(
        f"{run_dir.name}: fire arrival time\n(sensor labels = measured delay after FB18)"
    )
    axes.legend(loc="lower left", fontsize=8)
    figure.colorbar(mesh, ax=axes, label="arrival time after ignition (min)", shrink=0.8)
    figure.tight_layout()
    image_path = run_dir / "arrival_map.png"
    figure.savefig(image_path, dpi=130)
    plt.close(figure)
    return image_path


# ── Commands ──


def setup(args: argparse.Namespace) -> None:
    overrides = {
        "wind_speed_m_s": args.wind_speed,
        "wind_from_deg": args.wind_from,
        "fuel_height_m": args.fuel_height,
        "wind_height_m": args.wind_height,
        "end_time_s": args.end_time,
    }
    if args.steady_wind:
        overrides.update(wind_times_s=(), wind_speeds_m_s=(), wind_from_degs=())
    scenario = S5Scenario.from_rxcadre(
        args.chid, **{key: value for key, value in overrides.items() if value is not None}
    )
    print(write_case(scenario, args.out_dir))


def patchwork(args: argparse.Namespace) -> None:
    for fuel in patch_fuels(args.grass_height, separate_litter=args.separate_litter):
        print(
            f"{fuel.name:14s} FM{fuel.fuel_index:<2d} cover {fuel.cover_fraction:5.1%}  "
            f"load {fuel.load_kg_m2:.3f} kg/m2  height {fuel.height_m}"
        )
    for seed in args.seeds:
        chid = f"{args.chid}_s{seed}"
        write_patchwork_case(
            chid,
            seed=seed,
            patch_size_m=args.patch_size,
            grass_height_m=args.grass_height,
            separate_litter=args.separate_litter,
        )
        print(RUNS_DIR / chid)


def run(jobs: int) -> None:
    """Run every case under runs/ that has not completed yet, ``jobs`` at a time."""
    subprocess.run([str(BATCH_SCRIPT), str(RUNS_DIR), str(jobs)], check=True)


def compare(run_dirs: list[Path], *, detail: bool) -> None:
    for run_dir in run_dirs:
        result = compare_run(run_dir)
        print(
            f"{result.chid:28s} ROS model {result.model_ros_m_s:.3f} vs measured "
            f"{result.measured_ros_m_s:.3f} m/s | timing RMSE {result.rmse_s:5.1f} s | "
            f"implied ignition {result.implied_ignition_utc} UTC"
        )
        if detail:
            print(result.per_sensor.round(1).to_string())


def draw_maps(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        x, y, arrival = arrival_time_grid(run_dir)
        save_grid(run_dir, x, y, arrival)
        burned_m2 = np.isfinite(arrival).sum() * (x[1] - x[0]) * (y[1] - y[0])
        print(
            f"{run_dir.name}: burned ~{burned_m2:,.0f} m2, "
            f"last arrival {np.nanmax(arrival) / 60:.1f} min, "
            f"map {plot_map(run_dir, x, y, arrival)}"
        )


def calibrate() -> None:
    """Sweep fuel-bed depth with the 1-minute wind and the steady wind, run, then score."""
    cases = [
        S5Scenario.from_rxcadre(f"calw_h{depth_cm}cm", fuel_height_m=depth_cm / 100)
        for depth_cm in CALIBRATION_DEPTHS_CM
    ] + [
        S5Scenario.from_rxcadre(
            f"cal_h{depth_cm}cm_steady",
            fuel_height_m=depth_cm / 100,
            wind_times_s=(),
            wind_speeds_m_s=(),
            wind_from_degs=(),
        )
        for depth_cm in STEADY_CALIBRATION_DEPTHS_CM
    ]
    for scenario in cases:
        write_case(scenario)
    run(DEFAULT_PARALLEL_JOBS)
    compare([RUNS_DIR / scenario.chid for scenario in cases], detail=False)


def _run_dirs_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="*",
        default=[RUNS_DIR / "s5_base"],
        help="default: runs/s5_base",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup_parser = commands.add_parser("setup", help="write the S5 base case")
    setup_parser.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    setup_parser.add_argument("--chid", default="s5_base")
    setup_parser.add_argument("--wind-speed", type=float, help="override the measured wind speed")
    setup_parser.add_argument("--wind-from", type=float, help="override the measured direction")
    setup_parser.add_argument("--fuel-height", type=float, default=TUNED_FUEL_DEPTH_M)
    setup_parser.add_argument("--wind-height", type=float, default=None)
    setup_parser.add_argument("--end-time", type=float, default=2400.0)
    setup_parser.add_argument(
        "--steady-wind",
        action="store_true",
        help="one constant wind instead of the 1-minute series",
    )

    patchwork_parser = commands.add_parser("patchwork", help="write fuel-patchwork cases")
    patchwork_parser.add_argument("--chid", default="s5_patchwork")
    patchwork_parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    patchwork_parser.add_argument("--patch-size", type=float, default=DEFAULT_PATCH_SIZE_M)
    patchwork_parser.add_argument("--grass-height", type=float, default=TUNED_FUEL_DEPTH_M)
    patchwork_parser.add_argument("--separate-litter", action="store_true")

    run_parser = commands.add_parser("run", help="run every unfinished case under runs/")
    run_parser.add_argument("--jobs", type=int, default=DEFAULT_PARALLEL_JOBS)

    compare_parser = commands.add_parser("compare", help="score runs against the fire sensors")
    _run_dirs_argument(compare_parser)
    compare_parser.add_argument("--detail", action="store_true", help="print each sensor")

    _run_dirs_argument(commands.add_parser("map", help="arrival-time grid and map"))
    commands.add_parser("calibrate", help="sweep the fuel-bed depth")

    args = parser.parse_args()
    match args.command:
        case "setup":
            setup(args)
        case "patchwork":
            patchwork(args)
        case "run":
            run(args.jobs)
        case "compare":
            compare(args.run_dirs, detail=args.detail)
        case "map":
            draw_maps(args.run_dirs)
        case "calibrate":
            calibrate()


if __name__ == "__main__":
    main()
