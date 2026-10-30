"""Tabulate FDS's head-fire rate of spread over wind, moisture, fuel depth and fuel load.

Each table entry is a small controlled FDS run: uniform fuel, steady wind from
the west, and a full-width ignition line, so the front is planar and its
speed is the head-fire spread rate the level-set model assigns to that fuel and
wind. The table is the "true" potential spread-rate field for scenario replay in
wildfire_spread_prediction, and the target its drone-side estimator is fitted to.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import fdsreader
import numpy as np
import pandas as pd

from fire_sim.paths import PROJECT_ROOT

TABLE_RUNS_DIR = PROJECT_ROOT / "ros_table"
CELL_SIZE_M = 2.0
DOMAIN_LENGTH_M = 200.0
DOMAIN_WIDTH_M = 20.0
END_TIME_S = 3000.0
IGNITION_X_M = 10.0
SLICE_INTERVAL_S = 5.0
FIT_START_X_M = 30.0
FIT_END_X_M = 180.0

WIND_SPEEDS_M_S = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5)
DEAD_MOISTURES = (0.06, 0.08, 0.10)
FUEL_DEPTHS_M = (0.12, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
FUEL_LOADS_KG_M2 = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75)


@dataclass(frozen=True)
class TableCase:
    """One controlled run in the spread-rate table."""

    wind_speed_m_s: float
    dead_moisture: float
    fuel_depth_m: float
    fuel_load_kg_m2: float

    @property
    def chid(self) -> str:
        return (
            f"ros_w{self.wind_speed_m_s * 10:03.0f}_m{self.dead_moisture * 1000:03.0f}"
            f"_d{self.fuel_depth_m * 100:03.0f}_l{self.fuel_load_kg_m2 * 100:03.0f}"
        )


def all_cases() -> list[TableCase]:
    return [
        TableCase(*values)
        for values in itertools.product(
            WIND_SPEEDS_M_S, DEAD_MOISTURES, FUEL_DEPTHS_M, FUEL_LOADS_KG_M2
        )
    ]


def build_input(case: TableCase) -> str:
    """FDS input for one controlled planar-front run."""
    nx = round(DOMAIN_LENGTH_M / CELL_SIZE_M)
    ny = round(DOMAIN_WIDTH_M / CELL_SIZE_M)
    fuel = (
        f"VEG_LSET_FUEL_INDEX=1, VEG_LSET_SURF_LOAD={case.fuel_load_kg_m2:.3f}, "
        f"VEG_LSET_HT={case.fuel_depth_m:.3f}, VEG_LSET_M1={case.dead_moisture:.3f}"
    )
    x_ign = IGNITION_X_M
    lines = [
        f"&HEAD CHID='{case.chid}', TITLE='FDS head-fire ROS table entry' /",
        f"&MESH IJK={nx},{ny},1, XB=0,{DOMAIN_LENGTH_M},0,{DOMAIN_WIDTH_M},0,6 /",
        f"&TIME T_END={END_TIME_S} /",
        "&MISC LEVEL_SET_MODE=1 /",
        "&RADI RADIATION=F /",
        f"&WIND DIRECTION=270, SPEED={case.wind_speed_m_s} /",
        f"&DUMP DT_SLCF={SLICE_INTERVAL_S} /",
        *[
            f"&VENT MB='{side}', SURF_ID='OPEN' /"
            for side in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMAX")
        ],
        f"&SURF ID='IGN', {fuel}, VEG_LSET_IGNITE_TIME=0. /",
        "&SURF ID='BARE', DEFAULT=T /",
        f"&SURF ID='GRASS', {fuel} /",
        f"&VENT XB={x_ign},{x_ign + CELL_SIZE_M},0,{DOMAIN_WIDTH_M},0,0, SURF_ID='IGN' /",
        f"&VENT XB={x_ign + CELL_SIZE_M},{DOMAIN_LENGTH_M},0,{DOMAIN_WIDTH_M},0,0, "
        "SURF_ID='GRASS' /",
        "&SLCF AGL_SLICE=0.1, QUANTITY='LEVEL SET VALUE' /",
        "&TAIL /",
    ]
    return "\n".join(lines) + "\n"


def measured_head_ros(case_dir: Path) -> float:
    """Head-fire speed: slope of the centre-line front position over time between two fit stations.

    FDS's FIRE ARRIVAL TIME devices are unreliable when only written at the end of a run, so the
    front is tracked from the level-set slice instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level_set = fdsreader.Simulation(str(case_dir)).slices[0]
        frames = level_set[0].data
    times = np.asarray(level_set.times)
    centre_row = frames[:, :, frames.shape[2] // 2] > 0
    node_x = np.linspace(0.0, DOMAIN_LENGTH_M, centre_row.shape[1])
    front_x = np.array([node_x[burned].max() if burned.any() else np.nan for burned in centre_row])
    in_window = (front_x >= FIT_START_X_M) & (front_x <= FIT_END_X_M)
    if in_window.sum() >= 3 and np.ptp(front_x[in_window]) > 0:
        return float(np.polyfit(times[in_window], front_x[in_window], 1)[0])
    reached = np.isfinite(front_x)
    if not reached.any():
        return 0.0
    return float((np.nanmax(front_x) - IGNITION_X_M - CELL_SIZE_M) / times[-1])


def write_cases(cases: list[TableCase], runs_dir: Path) -> None:
    for case in cases:
        case_dir = runs_dir / case.chid
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"{case.chid}.fds").write_text(build_input(case))
        (case_dir / "case.json").write_text(json.dumps(asdict(case)))


def collect_table(runs_dir: Path, out_path: Path) -> pd.DataFrame:
    """Gather measured spread rates into a regular 4-D table saved as .npz (+ a CSV)."""
    rows = []
    for case in all_cases():
        case_dir = runs_dir / case.chid
        rows.append({**asdict(case), "head_ros_m_s": measured_head_ros(case_dir)})
    table = pd.DataFrame(rows)
    shape = (len(WIND_SPEEDS_M_S), len(DEAD_MOISTURES), len(FUEL_DEPTHS_M), len(FUEL_LOADS_KG_M2))
    ros = table.head_ros_m_s.to_numpy().reshape(shape)
    np.savez_compressed(
        out_path,
        wind_speed_m_s=np.array(WIND_SPEEDS_M_S),
        dead_moisture=np.array(DEAD_MOISTURES),
        fuel_depth_m=np.array(FUEL_DEPTHS_M),
        fuel_load_kg_m2=np.array(FUEL_LOADS_KG_M2),
        head_ros_m_s=ros,
    )
    table.to_csv(out_path.with_suffix(".csv"), index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write", help="write all table-case FDS inputs")
    collect = sub.add_parser("collect", help="measure spread rates from finished runs")
    collect.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "write":
        cases = all_cases()
        write_cases(cases, TABLE_RUNS_DIR)
        print(f"wrote {len(cases)} cases to {TABLE_RUNS_DIR}")
    else:
        table = collect_table(TABLE_RUNS_DIR, args.out)
        print(table.describe().round(4).to_string())


if __name__ == "__main__":
    main()
