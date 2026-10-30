"""Convert one finished burn-scenario FDS run into a compact training sample and drop raw output.

Each ``<id>.npz`` holds, on the 250 x 250 grid of 8 m cells (row = y north, col = x east):

- ``arrival_s``    fire arrival time (NaN = never burned) — the spread pattern to learn
- ``ignition_s``   time each cell was lit by hand (NaN = not lit) — a known model input
- ``fuel_depth``   m, ``fuel_load`` kg/m^2 (0 = unburnable / outside the block)
- ``wind_times_s``, ``wind_speed_m_s``, ``wind_from_deg`` — the wind series FDS used
- ``spec_json``    the full BurnSpec
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import fdsreader
import numpy as np

from fire_sim.burn_scenarios import (
    CELL_SIZE_M,
    BurnSpec,
    fuel_depth_field,
    ignition_times,
    wind_ramps,
)
from fire_sim.paths import PROJECT_ROOT

BURN_DATASET_DIR = PROJECT_ROOT / "burn_dataset"
RAW_OUTPUT_SUFFIXES = (".sf", ".ter", ".smv", ".out", ".bnd", ".csv", ".txt", ".pickle")


def arrival_at_cell_centres(case_dir: Path) -> np.ndarray:
    """First slice time the level set turns positive, averaged from nodes onto cell centres."""
    with warnings.catch_warnings(), _quiet_fdsreader():
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


@contextlib.contextmanager
def _quiet_fdsreader():
    """fdsreader prints tracebacks for vent tables with thousands of VENTs; we never read vents."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def export_run(case_dir: Path, out_dir: Path) -> Path:
    spec = BurnSpec(**json.loads((case_dir / "spec.json").read_text()))
    depth = fuel_depth_field(spec)
    times, speed_multiplier, direction = wind_ramps(spec)
    out_path = out_dir / f"{spec.scenario_id}.npz"
    np.savez_compressed(
        out_path,
        arrival_s=arrival_at_cell_centres(case_dir).astype(np.float32),
        ignition_s=ignition_times(spec).astype(np.float32),
        fuel_depth=depth.astype(np.float32),
        fuel_load=(spec.fuel_load_kg_m2 * depth / spec.fuel_depth_m).astype(np.float32),
        wind_times_s=times,
        wind_speed_m_s=spec.wind_speed_m_s * speed_multiplier,
        wind_from_deg=direction,
        cell_size_m=np.float32(CELL_SIZE_M),
        spec_json=np.array(json.dumps(asdict(spec))),
    )
    return out_path


def drop_raw_output(case_dir: Path) -> None:
    """Keep only the FDS input, spec and run log; everything else can be regenerated."""
    for path in case_dir.iterdir():
        if path.suffix in RAW_OUTPUT_SUFFIXES and path.name != "run.log":
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=BURN_DATASET_DIR)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_run(args.case_dir, args.out_dir)
    if not args.keep_raw:
        drop_raw_output(args.case_dir)


if __name__ == "__main__":
    main()
