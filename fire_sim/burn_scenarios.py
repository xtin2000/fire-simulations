"""Sample strip-ignited prescribed-burn scenarios on a 2 km landscape and write FDS inputs.

These are the Sim2Real training simulations. They mimic the operational RxCADRE
burns (L1G, L2G, L2F): a burn block surrounded by bare firebreak, lit in parallel
strips by drip-torch ATVs, in grass or longleaf-pine litter fuels. Following the
Sim2Real-Fire paper, uncertain inputs are randomised widely (domain randomisation)
rather than tuned to the real burns, which are reserved for testing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.ndimage import gaussian_filter

from fire_sim.paths import PROJECT_ROOT

BURN_SCENARIOS_DIR = PROJECT_ROOT / "burn_scenarios"
CELL_SIZE_M = 8.0
GRID_CELLS = 250
DOMAIN_SIZE_M = GRID_CELLS * CELL_SIZE_M
END_TIME_S = 9000.0
SLICE_INTERVAL_S = 30.0
IGNITION_TIME_STEP_S = 30.0
FUEL_DEPTH_CLASSES = 4


class FuelType(Enum):
    GRASS = "grass"
    LITTER = "litter"


@dataclass
class BurnSpec:
    """Everything needed to rebuild one burn scenario deterministically."""

    scenario_id: str
    seed: int
    fuel_type: str
    fuel_model_index: int
    fuel_depth_m: float
    fuel_load_kg_m2: float
    dead_moisture: float
    live_moisture: float
    patch_strength: float
    patch_length_m: float
    unburnable_fraction: float
    wind_speed_m_s: float
    wind_from_deg: float
    wind_shift_deg: float
    wind_shift_start_s: float
    gust_amplitude: float
    block_corners_m: list[list[float]]
    ignition_kind: str
    lighting_minutes: float
    line_on_upwind_edge: bool
    strip_spacing_m: float
    strip_angle_offset_deg: float
    atv_speed_m_s: float


def sample_burn(index: int, base_seed: int) -> BurnSpec:
    seed = base_seed + index
    rng = np.random.default_rng(seed)
    fuel_type = FuelType.GRASS if rng.random() < 0.6 else FuelType.LITTER
    if fuel_type is FuelType.GRASS:
        model, depth = 1, rng.uniform(0.2, 0.6)
        load, moisture = rng.uniform(0.2, 0.5), rng.uniform(0.05, 0.11)
    else:
        model = int(rng.choice([2, 9]))
        depth = rng.uniform(0.06, 0.5) if model == 9 else rng.uniform(0.2, 0.6)
        load, moisture = rng.uniform(0.3, 0.9), rng.uniform(0.06, 0.14)
    kind = rng.choice(["strips", "line", "point"], p=[0.8, 0.1, 0.1])
    return BurnSpec(
        scenario_id=f"burn{index:05d}",
        seed=seed,
        fuel_type=fuel_type.value,
        fuel_model_index=model,
        fuel_depth_m=round(float(depth), 3),
        fuel_load_kg_m2=round(float(load), 3),
        dead_moisture=round(float(moisture), 3),
        live_moisture=round(float(rng.uniform(0.8, 1.2)), 2),
        patch_strength=round(float(rng.uniform(0.0, 0.4)), 3),
        patch_length_m=round(float(rng.uniform(16.0, 120.0)), 1),
        unburnable_fraction=round(float(rng.uniform(0.0, 0.1)), 3),
        wind_speed_m_s=round(float(rng.uniform(0.5, 6.0)), 2),
        wind_from_deg=round(float(rng.uniform(0.0, 360.0)), 1),
        wind_shift_deg=round(float(rng.uniform(-45.0, 45.0)), 1),
        wind_shift_start_s=round(float(rng.uniform(600.0, 6000.0)), 0),
        gust_amplitude=round(float(rng.uniform(0.0, 0.3)), 3),
        block_corners_m=_random_block(rng),
        ignition_kind=str(kind),
        lighting_minutes=round(float(rng.uniform(15.0, 80.0)), 1),
        line_on_upwind_edge=bool(rng.random() < 0.5),
        strip_spacing_m=round(float(rng.uniform(24.0, 80.0)), 1),
        strip_angle_offset_deg=round(float(rng.uniform(-30.0, 30.0)), 1),
        atv_speed_m_s=round(float(rng.uniform(1.5, 4.0)), 2),
    )


def _random_block(rng: np.random.Generator) -> list[list[float]]:
    """A convex quadrilateral burn block of roughly 60-250 ha, inset from the domain edge."""
    centre = DOMAIN_SIZE_M / 2 + rng.uniform(-150.0, 150.0, 2)
    half_sizes = rng.uniform(350.0, 750.0, 2)
    rotation = rng.uniform(0.0, np.pi)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * half_sizes
    corners += rng.uniform(-0.15, 0.15, corners.shape) * half_sizes
    rotation_matrix = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    corners = corners @ rotation_matrix.T + centre
    return np.clip(corners, 60.0, DOMAIN_SIZE_M - 60.0).round(1).tolist()


def cell_centres() -> tuple[np.ndarray, np.ndarray]:
    centres = (np.arange(GRID_CELLS) + 0.5) * CELL_SIZE_M
    return np.meshgrid(centres, centres)


def block_mask(spec: BurnSpec) -> np.ndarray:
    xx, yy = cell_centres()
    inside = PolygonPath(np.array(spec.block_corners_m)).contains_points(
        np.c_[xx.ravel(), yy.ravel()]
    )
    return inside.reshape(xx.shape)


def fuel_depth_field(spec: BurnSpec) -> np.ndarray:
    """Patchy fuel depth [row=y, col=x] in metres; 0 outside the block and in unburnable spots."""
    rng = np.random.default_rng(spec.seed + 10_000)
    noise = gaussian_filter(
        rng.normal(size=(GRID_CELLS, GRID_CELLS)), spec.patch_length_m / CELL_SIZE_M
    )
    noise -= noise.mean()
    noise /= noise.std() or 1.0
    multiplier = np.clip(1.0 + spec.patch_strength * noise, 0.3, 2.0)
    edges = np.quantile(multiplier, np.linspace(0, 1, FUEL_DEPTH_CLASSES + 1)[1:-1])
    classes = np.digitize(multiplier, edges)
    class_means = np.array(
        [
            multiplier[classes == k].mean() if (classes == k).any() else 1.0
            for k in range(FUEL_DEPTH_CLASSES)
        ]
    )
    depth = spec.fuel_depth_m * class_means[classes]

    holes = gaussian_filter(rng.normal(size=(GRID_CELLS, GRID_CELLS)), 2.0)
    depth[holes > np.quantile(holes, 1.0 - spec.unburnable_fraction)] = 0.0
    depth[~block_mask(spec)] = 0.0
    return depth


def wind_ramps(spec: BurnSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times, speed multipliers and from-directions: gusts every 120 s plus one gradual shift."""
    rng = np.random.default_rng(spec.seed + 20_000)
    times = np.arange(0.0, END_TIME_S + 1.0, 120.0)
    gusts = 1.0 + spec.gust_amplitude * gaussian_filter(rng.normal(size=times.size), 1.0) * 2.0
    shift_progress = np.clip((times - spec.wind_shift_start_s) / 1200.0, 0.0, 1.0)
    direction = (spec.wind_from_deg + spec.wind_shift_deg * shift_progress) % 360.0
    return times, np.clip(gusts, 0.3, 1.7), direction


def ignition_times(spec: BurnSpec) -> np.ndarray:
    """Ignition time (s) per cell [row, col]; NaN where never lit by hand."""
    lit = np.full((GRID_CELLS, GRID_CELLS), np.nan)
    inside = block_mask(spec) & (fuel_depth_field(spec) > 0)
    rng = np.random.default_rng(spec.seed + 30_000)
    toward = np.radians(spec.wind_from_deg + 180.0)
    downwind = np.array([np.sin(toward), np.cos(toward)])
    xx, yy = cell_centres()
    centre = np.array(spec.block_corners_m).mean(axis=0)

    if spec.ignition_kind == "point":
        start = centre - downwind * rng.uniform(100.0, 300.0)
        distance = np.hypot(xx - start[0], yy - start[1])
        nearest = (distance < CELL_SIZE_M * 1.5) & inside
        if not nearest.any() and inside.any():
            # The target spot is bare or outside the block: light the closest burnable cell.
            nearest = distance == distance[inside].min()
            nearest &= inside
        lit[nearest] = 0.0
        return lit

    strip_normal_angle = toward + np.radians(spec.strip_angle_offset_deg)
    normal = np.array([np.sin(strip_normal_angle), np.cos(strip_normal_angle)])
    along_strip = np.array([normal[1], -normal[0]])
    across = (xx - centre[0]) * normal[0] + (yy - centre[1]) * normal[1]
    along = (xx - centre[0]) * along_strip[0] + (yy - centre[1]) * along_strip[1]
    on_any_strip = (
        np.abs(((across / spec.strip_spacing_m) + 0.5) % 1.0 - 0.5) * spec.strip_spacing_m
    )
    near_line = on_any_strip < CELL_SIZE_M * 0.6

    if spec.ignition_kind == "line":
        if not inside.any():
            return lit
        first_strip = (
            across[inside].min() + spec.strip_spacing_m
            if spec.line_on_upwind_edge
            else across[inside].max() - spec.strip_spacing_m
        )
        line = near_line & (np.abs(across - first_strip) < spec.strip_spacing_m / 2) & inside
        if line.any():
            lit[line] = np.abs(along[line] - along[line].min()) / spec.atv_speed_m_s
        return lit

    # Strip firing: start on the downwind edge and work upwind; ATVs light adjacent strips in
    # parallel, then shift together to the next group, pausing between strips. The crew size
    # is whatever finishes lighting in the sampled duration (real RxCADRE burns: 17-65 min).
    strip_index = np.round(across / spec.strip_spacing_m).astype(int)
    strips = sorted(np.unique(strip_index[near_line & inside]), reverse=True)
    strip_length = (along[inside].max() - along[inside].min()) if inside.any() else 0.0
    pass_duration = strip_length / spec.atv_speed_m_s + rng.uniform(30.0, 120.0)
    atv_count = atv_crew_size(len(strips), pass_duration, spec.lighting_minutes)
    for position, strip in enumerate(strips):
        group = position // atv_count
        on_strip = near_line & inside & (strip_index == strip)
        if not on_strip.any():
            continue
        reverse = (group % 2) == 1
        distance = along[on_strip] - along[inside].min()
        if reverse:
            distance = strip_length - distance
        lit[on_strip] = group * pass_duration + distance / spec.atv_speed_m_s
    return lit


def atv_crew_size(strip_count: int, pass_duration_s: float, lighting_minutes: float) -> int:
    """ATVs needed to light all strips within the target lighting time (at least 1, at most 12)."""
    groups_possible = max(1.0, lighting_minutes * 60.0 / pass_duration_s)
    return int(np.clip(np.ceil(strip_count / groups_possible), 1, 12))


def build_input(spec: BurnSpec) -> str:
    depth = fuel_depth_field(spec)
    lit = ignition_times(spec)
    times, speed_multiplier, direction = wind_ramps(spec)
    lines = [
        f"&HEAD CHID='{spec.scenario_id}', "
        f"TITLE='Strip-ignited burn scenario {spec.scenario_id}' /",
        f"&MESH IJK={GRID_CELLS},{GRID_CELLS},1, XB=0,{DOMAIN_SIZE_M},0,{DOMAIN_SIZE_M},0,16 /",
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
        "&SURF ID='BARE', DEFAULT=T /",
    ]
    lit_times = np.where(
        np.isfinite(lit), np.round(lit / IGNITION_TIME_STEP_S) * IGNITION_TIME_STEP_S, np.nan
    )
    lines += _surface_lines(spec, depth, lit_times)
    lines += ["&SLCF AGL_SLICE=1, QUANTITY='LEVEL SET VALUE' /", "&TAIL /"]
    return "\n".join(lines) + "\n"


def _fuel_params(spec: BurnSpec, depth_m: float) -> str:
    load = spec.fuel_load_kg_m2 * depth_m / spec.fuel_depth_m
    return (
        f"VEG_LSET_FUEL_INDEX={spec.fuel_model_index}, VEG_LSET_SURF_LOAD={load:.4f}, "
        f"VEG_LSET_HT={depth_m:.3f}, VEG_LSET_M1={spec.dead_moisture:.3f}, "
        f"VEG_LSET_M10={spec.dead_moisture + 0.01:.3f}, "
        f"VEG_LSET_M100={spec.dead_moisture + 0.02:.3f}, "
        f"VEG_LSET_MLH={spec.live_moisture:.2f}, VEG_LSET_MLW={spec.live_moisture:.2f}"
    )


def _surface_lines(spec: BurnSpec, depth: np.ndarray, lit_times: np.ndarray) -> list[str]:
    """One SURF per (fuel class, ignition time) combination and row-run VENTs for each."""
    keys = np.where(depth > 0, depth, -1.0).astype(object)
    lines: list[str] = []
    surf_ids: dict[tuple[float, float], str] = {}
    for row in range(GRID_CELLS):
        for col in range(GRID_CELLS):
            if depth[row, col] <= 0:
                continue
            key = (
                round(float(depth[row, col]), 4),
                float(lit_times[row, col]) if np.isfinite(lit_times[row, col]) else -1.0,
            )
            if key not in surf_ids:
                surf_id = f"F{len(surf_ids)}"
                surf_ids[key] = surf_id
                extra = f", VEG_LSET_IGNITE_TIME={key[1]:.0f}" if key[1] >= 0 else ""
                lines.append(f"&SURF ID='{surf_id}', {_fuel_params(spec, key[0])}{extra} /")
            keys[row, col] = surf_ids[key]
    for row in range(GRID_CELLS):
        y0 = row * CELL_SIZE_M
        run_start = 0
        for col in range(1, GRID_CELLS + 1):
            if col < GRID_CELLS and keys[row, col] == keys[row, run_start]:
                continue
            surf_id = keys[row, run_start]
            if isinstance(surf_id, str):
                lines.append(
                    f"&VENT XB={run_start * CELL_SIZE_M:g},{col * CELL_SIZE_M:g},{y0:g},"
                    f"{y0 + CELL_SIZE_M:g},0,0, SURF_ID='{surf_id}' /"
                )
            run_start = col
    return lines


def write_scenarios(count: int, base_seed: int, out_dir: Path, *, start: int = 0) -> None:
    for index in range(start, start + count):
        spec = sample_burn(index, base_seed)
        case_dir = out_dir / spec.scenario_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"{spec.scenario_id}.fds").write_text(build_input(spec))
        (case_dir / "spec.json").write_text(json.dumps(asdict(spec), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--out-dir", type=Path, default=BURN_SCENARIOS_DIR)
    args = parser.parse_args()
    write_scenarios(args.count, args.seed, args.out_dir, start=args.start)
    print(f"wrote scenarios {args.start}..{args.start + args.count - 1} to {args.out_dir}")


if __name__ == "__main__":
    main()
