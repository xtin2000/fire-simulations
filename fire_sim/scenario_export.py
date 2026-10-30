"""Convert finished scenario FDS runs into ground-truth files for wildfire_spread_prediction.

Each scenario becomes one ``<id>.npz`` with arrays on the TerrainGrid layout
([row=y, col=x], 2 m cells, origin lower-left):

- ``arrival_s``      fire arrival time at each cell centre (NaN = never burned)
- ``ros_vector``     (H, W, 2) spread velocity (m/s, x east / y north) from grad(arrival)
- ``fuel_load``      kg/m^2 per cell (0 on bare ground / firebreak)
- ``fuel_depth``     m per cell
- ``moisture``       dead fuel moisture fraction per cell
- ``wind_times_s``, ``wind_speed_m_s``, ``wind_from_deg``   the scenario's wind time series
- ``spec_json``      the ScenarioSpec as JSON text
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import fields
from pathlib import Path

import fdsreader
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from fire_sim.scenarios import CELL_SIZE_M, ScenarioSpec, fuel_depth_field, wind_ramps

MAX_REPORTED_ROS_M_S = 2.0


def load_spec(case_dir: Path) -> ScenarioSpec:
    raw = json.loads((case_dir / "spec.json").read_text())
    return ScenarioSpec(**{f.name: raw[f.name] for f in fields(ScenarioSpec)})


def arrival_at_cell_centres(case_dir: Path, chid: str) -> np.ndarray:
    """First time the level set turns positive, averaged from slice nodes onto cell centres."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level_set = fdsreader.Simulation(str(case_dir)).slices.filter_by_quantity(
            "LEVEL SET VALUE"
        )[0]
        frames = level_set[0].data
    times = np.asarray(level_set.times)
    is_burned = frames > 0
    node_arrival = np.where(is_burned.any(axis=0), times[is_burned.argmax(axis=0)], np.nan).T
    corners = np.stack(
        [
            node_arrival[:-1, :-1],
            node_arrival[1:, :-1],
            node_arrival[:-1, 1:],
            node_arrival[1:, 1:],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(corners, axis=0)


def spread_velocity(arrival_s: np.ndarray) -> np.ndarray:
    """Spread velocity = grad(t) / |grad(t)|^2, from a lightly smoothed arrival field."""
    burned = np.isfinite(arrival_s)
    filled = np.where(burned, arrival_s, np.nanmax(arrival_s) if burned.any() else 0.0)
    smoothed = gaussian_filter(filled, sigma=1.0)
    grad_y, grad_x = np.gradient(smoothed, CELL_SIZE_M)
    grad_squared = grad_x**2 + grad_y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.stack([grad_x / grad_squared, grad_y / grad_squared], axis=-1)
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    velocity = np.where(
        speed > MAX_REPORTED_ROS_M_S, velocity * MAX_REPORTED_ROS_M_S / speed, velocity
    )
    velocity[~burned] = 0.0
    return np.nan_to_num(velocity).astype(np.float32)


def export_case(case_dir: Path, out_dir: Path) -> dict[str, float | str]:
    spec = load_spec(case_dir)
    arrival = arrival_at_cell_centres(case_dir, spec.scenario_id)
    depth = fuel_depth_field(spec)
    times, speed_multiplier, direction = wind_ramps(spec)
    np.savez_compressed(
        out_dir / f"{spec.scenario_id}.npz",
        arrival_s=arrival.astype(np.float32),
        ros_vector=spread_velocity(arrival),
        fuel_load=(spec.fuel_load_kg_m2 * depth / spec.mean_fuel_depth_m).astype(np.float32),
        fuel_depth=depth.astype(np.float32),
        moisture=np.full_like(depth, spec.dead_moisture, dtype=np.float32),
        wind_times_s=times,
        wind_speed_m_s=spec.wind_speed_m_s * speed_multiplier,
        wind_from_deg=direction,
        cell_size_m=np.float32(CELL_SIZE_M),
        spec_json=np.array(json.dumps(spec.__dict__)),
    )
    burned = np.isfinite(arrival)
    return {
        **spec.__dict__,
        "burned_area_m2": float(burned.sum() * CELL_SIZE_M**2),
        "last_arrival_s": float(np.nanmax(arrival)) if burned.any() else np.nan,
        "reaches_edge": bool(
            burned[0].any() or burned[-1].any() or burned[:, 0].any() or burned[:, -1].any()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in sorted(args.case_dirs):
        log = case_dir / "run.log"
        if not log.exists() or "completed successfully" not in log.read_text():
            print(f"skip {case_dir.name}: run did not complete")
            continue
        rows.append(export_case(case_dir, args.out_dir))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    print(f"exported {len(rows)} scenarios -> {args.out_dir}")


if __name__ == "__main__":
    main()
