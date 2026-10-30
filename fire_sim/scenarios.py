"""Sample fictitious RxCADRE-S5-like fire scenarios on a 400 m landscape and write FDS inputs.

Fuel, moisture and wind ranges come from the RxCADRE 2012 small-plot (S3-S9)
measurements; the fuel-depth anchor (0.37 m) is the S5 calibration from
s5_fds_fast.py. The landscape grid matches the wildfire_spread_prediction
TerrainGrid: 200 x 200 cells of 2 m, origin at the lower-left corner, x east,
y north.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from fire_sim.paths import SCENARIOS_DIR

GRID_CELLS = 200
CELL_SIZE_M = 2.0
DOMAIN_SIZE_M = GRID_CELLS * CELL_SIZE_M
DOMAIN_TOP_M = 6.0
END_TIME_S = 3600.0
SLICE_INTERVAL_S = 5.0

CALIBRATED_FUEL_DEPTH_M = 0.37
S5_MEAN_GRASS_HEIGHT_CM = 11.88
FUEL_DEPTH_CLASSES = 5
IGNITER_WALK_SPEED_M_S = 1.34


class IgnitionKind(Enum):
    POINT = "point"
    LINE = "line"


@dataclass
class ScenarioSpec:
    """Everything needed to rebuild one fictitious scenario deterministically."""

    scenario_id: str
    seed: int
    wind_speed_m_s: float
    wind_from_deg: float
    wind_shift_deg: float
    wind_shift_start_s: float
    gust_amplitude: float
    dead_moisture: float
    fuel_load_kg_m2: float
    mean_fuel_depth_m: float
    patch_strength: float
    patch_length_m: float
    has_firebreak: bool
    ignition_kind: str
    ignition_x_m: float
    ignition_y_m: float


def sample_scenario(index: int, base_seed: int) -> ScenarioSpec:
    """Draw one scenario from the RxCADRE small-plot ranges."""
    seed = base_seed + index
    rng = np.random.default_rng(seed)
    wind_from = float((335.0 + rng.uniform(-45.0, 45.0)) % 360.0)
    ignition_x, ignition_y = _upwind_ignition_point(wind_from, rng)
    grass_height_cm = rng.uniform(9.8, 14.1)
    return ScenarioSpec(
        scenario_id=f"scn{index:03d}",
        seed=seed,
        wind_speed_m_s=round(float(rng.uniform(1.5, 4.5)), 2),
        wind_from_deg=round(wind_from, 1),
        wind_shift_deg=round(float(rng.uniform(-40.0, 40.0)), 1),
        wind_shift_start_s=round(float(rng.uniform(300.0, 1500.0)), 0),
        gust_amplitude=round(float(rng.uniform(0.0, 0.25)), 3),
        dead_moisture=round(float(rng.uniform(0.064, 0.100)), 3),
        fuel_load_kg_m2=round(float(rng.uniform(0.24, 0.41)), 3),
        mean_fuel_depth_m=round(
            CALIBRATED_FUEL_DEPTH_M * grass_height_cm / S5_MEAN_GRASS_HEIGHT_CM, 3
        ),
        patch_strength=round(float(rng.uniform(0.0, 0.35)), 3),
        patch_length_m=round(float(rng.uniform(8.0, 40.0)), 1),
        has_firebreak=bool(rng.random() < 0.3),
        ignition_kind=(IgnitionKind.LINE if rng.random() < 0.5 else IgnitionKind.POINT).value,
        ignition_x_m=round(ignition_x, 1),
        ignition_y_m=round(ignition_y, 1),
    )


def _upwind_ignition_point(wind_from_deg: float, rng: np.random.Generator) -> tuple[float, float]:
    """Place ignition upwind of the centre so the head fire has room to run."""
    toward = np.radians(wind_from_deg + 180.0)
    downwind = np.array([np.sin(toward), np.cos(toward)])
    centre = np.full(2, DOMAIN_SIZE_M / 2)
    point = centre - 110.0 * downwind + rng.uniform(-40.0, 40.0, 2)
    return tuple(np.clip(point, 40.0, DOMAIN_SIZE_M - 40.0).tolist())


def fuel_depth_field(spec: ScenarioSpec) -> np.ndarray:
    """Spatially correlated fuel depth [row=y, col=x] in metres, quantised to a few classes."""
    rng = np.random.default_rng(spec.seed + 10_000)
    noise = gaussian_filter(
        rng.normal(size=(GRID_CELLS, GRID_CELLS)), spec.patch_length_m / CELL_SIZE_M
    )
    noise -= noise.mean()
    noise /= noise.std() or 1.0
    multiplier = np.clip(1.0 + spec.patch_strength * noise, 0.4, 1.8)
    class_edges = np.quantile(multiplier, np.linspace(0, 1, FUEL_DEPTH_CLASSES + 1)[1:-1])
    classes = np.digitize(multiplier, class_edges)
    class_means = np.array([multiplier[classes == k].mean() for k in range(FUEL_DEPTH_CLASSES)])
    depth = spec.mean_fuel_depth_m * class_means[classes]
    if spec.has_firebreak:
        depth[_firebreak_mask(spec)] = 0.0
    return depth


def _firebreak_mask(spec: ScenarioSpec) -> np.ndarray:
    """A 6 m bare road crossing the domain downwind of the ignition, roughly across the wind."""
    toward = np.radians(spec.wind_from_deg + 180.0)
    downwind = np.array([np.sin(toward), np.cos(toward)])
    road_centre = np.array([spec.ignition_x_m, spec.ignition_y_m]) + 140.0 * downwind
    centres = (np.arange(GRID_CELLS) + 0.5) * CELL_SIZE_M
    xx, yy = np.meshgrid(centres, centres)
    along_wind = (xx - road_centre[0]) * downwind[0] + (yy - road_centre[1]) * downwind[1]
    return np.abs(along_wind) < 3.0


def wind_ramps(spec: ScenarioSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times, speed multipliers and from-directions: gusts every 60 s plus one gradual shift."""
    rng = np.random.default_rng(spec.seed + 20_000)
    times = np.arange(0.0, END_TIME_S + 1.0, 60.0)
    gusts = 1.0 + spec.gust_amplitude * gaussian_filter(rng.normal(size=times.size), 1.0) * 2.0
    speed_multiplier = np.clip(gusts, 0.4, 1.6)
    shift_progress = np.clip((times - spec.wind_shift_start_s) / 600.0, 0.0, 1.0)
    direction = (spec.wind_from_deg + spec.wind_shift_deg * shift_progress) % 360.0
    return times, speed_multiplier, direction


def build_input(spec: ScenarioSpec) -> str:
    """Full FDS level-set input text for one scenario."""
    times, speed_multiplier, direction = wind_ramps(spec)
    depth = fuel_depth_field(spec)
    lines = [
        f"&HEAD CHID='{spec.scenario_id}', "
        f"TITLE='Fictitious RxCADRE-S5-like scenario {spec.scenario_id}' /",
        f"&MESH IJK={GRID_CELLS},{GRID_CELLS},1, "
        f"XB=0,{DOMAIN_SIZE_M},0,{DOMAIN_SIZE_M},0,{DOMAIN_TOP_M} /",
        f"&TIME T_END={END_TIME_S} /",
        "&MISC LEVEL_SET_MODE=1 /",
        "&RADI RADIATION=F /",
        f"&WIND SPEED={spec.wind_speed_m_s}, DIRECTION={spec.wind_from_deg}, "
        "RAMP_SPEED_T='wind speed', RAMP_DIRECTION_T='wind dir' /",
        *[
            f"&RAMP ID='wind speed', T={t:.0f}, F={f:.3f} /"
            for t, f in zip(times, speed_multiplier, strict=True)
        ],
        *[
            f"&RAMP ID='wind dir', T={t:.0f}, F={f:.1f} /"
            for t, f in zip(times, direction, strict=True)
        ],
        f"&DUMP DT_SLCF={SLICE_INTERVAL_S} /",
        *[
            f"&VENT MB='{side}', SURF_ID='OPEN' /"
            for side in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMAX")
        ],
        "&SURF ID='BARE', DEFAULT=T, COLOR='GRAY 60' /",
    ]
    ignition_cells = _ignition_cells(spec)
    lines += _ignition_lines(spec, ignition_cells)
    lines += _fuel_lines(spec, depth, ignition_cells)
    lines += ["&SLCF AGL_SLICE=0.1, QUANTITY='LEVEL SET VALUE' /", "&TAIL /"]
    return "\n".join(lines) + "\n"


def _fuel_surf(surf_id: str, spec: ScenarioSpec, depth_m: float, extra: str = "") -> str:
    load = spec.fuel_load_kg_m2 * depth_m / spec.mean_fuel_depth_m
    text = (
        f"&SURF ID='{surf_id}', VEG_LSET_FUEL_INDEX=1, VEG_LSET_SURF_LOAD={load:.4f}, "
        f"VEG_LSET_HT={depth_m:.3f}, VEG_LSET_M1={spec.dead_moisture:.3f}"
    )
    return text + (f", {extra}" if extra else "") + " /"


def _ignition_cells(spec: ScenarioSpec) -> dict[tuple[int, int], float]:
    """Map of (row, col) -> ignition time for the point patch or walked drip-torch line."""
    centre_col = int(spec.ignition_x_m // CELL_SIZE_M)
    centre_row = int(spec.ignition_y_m // CELL_SIZE_M)
    if spec.ignition_kind == IgnitionKind.POINT.value:
        return {
            (centre_row + dr, centre_col + dc): 0.0 for dr in range(-1, 2) for dc in range(-1, 2)
        }
    toward = np.radians(spec.wind_from_deg + 180.0)
    across = np.array([np.cos(toward), -np.sin(toward)])
    cells: dict[tuple[int, int], float] = {}
    for offset_m in np.arange(-25.0, 25.0 + 0.1, CELL_SIZE_M / 2):
        x, y = np.array([spec.ignition_x_m, spec.ignition_y_m]) + offset_m * across
        key = (int(y // CELL_SIZE_M), int(x // CELL_SIZE_M))
        cells.setdefault(key, round(abs(offset_m) / IGNITER_WALK_SPEED_M_S, 1))
    return cells


def _ignition_lines(spec: ScenarioSpec, cells: dict[tuple[int, int], float]) -> list[str]:
    lines = []
    for ignite_time in sorted(set(cells.values())):
        surf_id = f"IGN t{ignite_time:g}"
        lines.append(
            _fuel_surf(
                surf_id,
                spec,
                spec.mean_fuel_depth_m,
                f"VEG_LSET_IGNITE_TIME={ignite_time}, COLOR='RED'",
            )
        )
        for (row, col), cell_time in cells.items():
            if cell_time == ignite_time:
                x0, y0 = col * CELL_SIZE_M, row * CELL_SIZE_M
                lines.append(
                    f"&VENT XB={x0:g},{x0 + CELL_SIZE_M:g},{y0:g},{y0 + CELL_SIZE_M:g},0,0, "
                    f"SURF_ID='{surf_id}' /"
                )
    return lines


def _fuel_lines(
    spec: ScenarioSpec, depth: np.ndarray, ignition_cells: dict[tuple[int, int], float]
) -> list[str]:
    """One SURF per fuel-depth class and one VENT per contiguous same-class run in each row."""
    class_depths = sorted({float(d) for d in np.unique(depth) if d > 0})
    surf_ids = {d: f"GRASS {k}" for k, d in enumerate(class_depths)}
    lines = [_fuel_surf(surf_ids[d], spec, d, "COLOR='GREEN'") for d in class_depths]
    for row in range(GRID_CELLS):
        row_depths = depth[row].copy()
        for ign_row, ign_col in ignition_cells:
            if ign_row == row and 0 <= ign_col < GRID_CELLS:
                row_depths[ign_col] = -1.0
        y0 = row * CELL_SIZE_M
        run_start = 0
        for col in range(1, GRID_CELLS + 1):
            if col < GRID_CELLS and row_depths[col] == row_depths[run_start]:
                continue
            run_depth = float(row_depths[run_start])
            if run_depth > 0:
                lines.append(
                    f"&VENT XB={run_start * CELL_SIZE_M:g},{col * CELL_SIZE_M:g},{y0:g},"
                    f"{y0 + CELL_SIZE_M:g},0,0, SURF_ID='{surf_ids[run_depth]}' /"
                )
            run_start = col
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=SCENARIOS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [sample_scenario(index, args.seed) for index in range(args.count)]
    for spec in specs:
        case_dir = args.out_dir / spec.scenario_id
        case_dir.mkdir(exist_ok=True)
        (case_dir / f"{spec.scenario_id}.fds").write_text(build_input(spec))
        (case_dir / "spec.json").write_text(json.dumps(asdict(spec), indent=2))
    print(f"wrote {len(specs)} scenarios to {args.out_dir}")


if __name__ == "__main__":
    main()
