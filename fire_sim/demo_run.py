"""A single, fully explained FDS run: one random ignition point on uniform grass.

Builds a small FDS input with comments, runs it, and compares the simulated spread with the
values FDS's own formulas predict (vege.f90): head, flank and backing speeds of the egg-shaped
fire. Uses the S5 grass and wind magnitude; the ignition point and wind direction are random.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fire_sim.paths import RUNS_DIR

DOMAIN_M = 400.0
CELL_M = 2.0
END_TIME_S = 1800.0
SLICE_EVERY_S = 10.0
FUEL = {"index": 1, "load": 0.282, "depth": 0.37, "moisture": 0.086}
WIND_SPEED_M_S = 2.8


@dataclass
class DemoInputs:
    seed: int
    ignition_x_m: float
    ignition_y_m: float
    wind_from_deg: float


def random_inputs(seed: int) -> DemoInputs:
    """Ignition anywhere in the middle half of the area; wind from any direction."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0.25 * DOMAIN_M, 0.75 * DOMAIN_M, 2)
    return DemoInputs(
        seed, round(float(x), 1), round(float(y), 1), round(float(rng.uniform(0, 360)), 1)
    )


def build_input(inputs: DemoInputs, chid: str) -> str:
    cells = round(DOMAIN_M / CELL_M)
    x0 = math.floor(inputs.ignition_x_m / CELL_M) * CELL_M
    y0 = math.floor(inputs.ignition_y_m / CELL_M) * CELL_M
    fuel = (
        f"VEG_LSET_FUEL_INDEX={FUEL['index']}, VEG_LSET_SURF_LOAD={FUEL['load']}, "
        f"VEG_LSET_HT={FUEL['depth']}, VEG_LSET_M1={FUEL['moisture']}"
    )
    return f"""Demo run: one random ignition point on uniform grass (seed {inputs.seed}).
FDS only reads lines that start with an ampersand, so all other lines are comments.

&HEAD CHID='{chid}', TITLE='Random ignition demo' /

Area: {DOMAIN_M:g} m x {DOMAIN_M:g} m, squares of {CELL_M:g} m ({cells} x {cells}).
x = metres east, y = metres north.
&MESH IJK={cells},{cells},1, XB=0,{DOMAIN_M:g},0,{DOMAIN_M:g},0,6 /

Simulate {END_TIME_S / 60:g} minutes, fire-front (level-set) mode only.
&TIME T_END={END_TIME_S:g} /
&MISC LEVEL_SET_MODE=1 /
&RADI RADIATION=F /

Wind: {WIND_SPEED_M_S} m/s blowing FROM {inputs.wind_from_deg} degrees (0 = north, 90 = east).
&WIND SPEED={WIND_SPEED_M_S}, DIRECTION={inputs.wind_from_deg} /

Open sides and top; save the fire edge every {SLICE_EVERY_S:g} s.
&VENT MB='XMIN', SURF_ID='OPEN' /
&VENT MB='XMAX', SURF_ID='OPEN' /
&VENT MB='YMIN', SURF_ID='OPEN' /
&VENT MB='YMAX', SURF_ID='OPEN' /
&VENT MB='ZMAX', SURF_ID='OPEN' /
&DUMP DT_SLCF={SLICE_EVERY_S:g} /

Fuel everywhere: short grass (type 1) with the S5 values: 0.282 kg/m2, 0.37 m deep, 8.6% moisture.
Ignition: the {CELL_M:g} m square containing the random point
({inputs.ignition_x_m}, {inputs.ignition_y_m}), lit at t = 0.
FDS rejects a fuel patch that overlaps another,
so the grass is laid in four pieces around the lit square.
&SURF ID='GRASS', {fuel} /
&SURF ID='LIT', {fuel}, VEG_LSET_IGNITE_TIME=0 /
&VENT XB={x0:g},{x0 + CELL_M:g},{y0:g},{y0 + CELL_M:g},0,0, SURF_ID='LIT' /
&VENT XB=0,{DOMAIN_M:g},0,{y0:g},0,0, SURF_ID='GRASS' /
&VENT XB=0,{DOMAIN_M:g},{y0 + CELL_M:g},{DOMAIN_M:g},0,0, SURF_ID='GRASS' /
&VENT XB=0,{x0:g},{y0:g},{y0 + CELL_M:g},0,0, SURF_ID='GRASS' /
&VENT XB={x0 + CELL_M:g},{DOMAIN_M:g},{y0:g},{y0 + CELL_M:g},0,0, SURF_ID='GRASS' /

Output: where the fire edge is (level-set value > 0 = burned).
&SLCF AGL_SLICE=1, QUANTITY='LEVEL SET VALUE' /
&TAIL /
"""


def predicted_speeds(ros_no_wind: float) -> dict[str, float]:
    """Head, back and sideways speeds from the formulas in vege.f90 (275-279, 424, 441, 825-845)."""
    depth, sigma_per_cm = FUEL["depth"], 11483.0 * 0.01
    beta = FUEL["load"] / depth / 512.0
    wind_reduction = 1.83 / math.log((20 + 1.18 * depth) / (0.43 * depth))
    flame_wind = WIND_SPEED_M_S * wind_reduction
    b = 0.15988 * sigma_per_cm**0.54
    c = 7.47 * math.exp(-0.8711 * sigma_per_cm**0.55)
    e = 0.715 * math.exp(-0.01094 * sigma_per_cm)
    beta_opt = 0.20395 * sigma_per_cm**-0.8189
    wind_boost = c * (3.281 * flame_wind * 60) ** b * (beta / beta_opt) ** (-e)
    head = ros_no_wind * (1 + wind_boost)
    length_to_breadth = max(
        1.0,
        min(
            8.0,
            0.936 * math.exp(0.2566 * flame_wind) + 0.461 * math.exp(-0.1548 * flame_wind) - 0.397,
        ),
    )
    root = math.sqrt(length_to_breadth**2 - 1)
    head_to_back = (length_to_breadth + root) / (length_to_breadth - root)
    back = head / head_to_back
    semi_major_rate = 0.5 * (head + back)
    semi_minor_rate = semi_major_rate / length_to_breadth
    centre_drift = semi_major_rate - back
    return {
        "no_wind_m_s": ros_no_wind,
        "wind_reduction": wind_reduction,
        "flame_wind_m_s": flame_wind,
        "wind_boost": wind_boost,
        "head_m_s": head,
        "back_m_s": back,
        # The ellipse's widest point drifts downwind, so sideways growth through the
        # ignition point is narrower than the semi-minor axis growth.
        "flank_m_s": semi_minor_rate * math.sqrt(1 - (centre_drift / semi_major_rate) ** 2),
        "widest_half_width_m_s": semi_minor_rate,
        "centre_drift_m_s": centre_drift,
        "length_to_breadth": length_to_breadth,
    }


def ros_no_wind_from_log(out_file: Path) -> float:
    text = out_file.read_text()
    block = text[text.index(" GRASS") :]
    return float(re.search(r"ROS_00 \(m/s\)\s+([0-9.E+-]+)", block).group(1))


def arrival_grid(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates (m) and first time each node's level set turned positive."""
    import contextlib
    import io
    import warnings

    import fdsreader

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        level_set = fdsreader.Simulation(str(run_dir)).slices[0]
        frames = level_set[0].data
    times = np.asarray(level_set.times)
    burned = frames > 0
    arrival = np.where(burned.any(axis=0), times[burned.argmax(axis=0)], np.nan).T
    nodes = np.linspace(0, DOMAIN_M, arrival.shape[0])
    return nodes, arrival


def measured_speeds(inputs: DemoInputs, nodes: np.ndarray, arrival: np.ndarray) -> dict[str, float]:
    """Spread speed along the downwind, upwind and crosswind lines through the ignition point."""
    toward = math.radians(inputs.wind_from_deg + 180.0)
    downwind = np.array([math.sin(toward), math.cos(toward)])
    crosswind = np.array([downwind[1], -downwind[0]])
    centre = (
        np.array(
            [
                math.floor(inputs.ignition_x_m / CELL_M) + 0.5,
                math.floor(inputs.ignition_y_m / CELL_M) + 0.5,
            ]
        )
        * CELL_M
    )

    def speed_along(direction: np.ndarray) -> float:
        distances = np.arange(10.0, 400.0, CELL_M)
        points = centre + distances[:, None] * direction
        inside = np.all((points >= 0) & (points <= DOMAIN_M), axis=1)
        cols = np.clip(np.round(points[inside, 0] / CELL_M).astype(int), 0, len(nodes) - 1)
        rows = np.clip(np.round(points[inside, 1] / CELL_M).astype(int), 0, len(nodes) - 1)
        times = arrival[rows, cols]
        reached = np.isfinite(times) & (times > 60)
        if reached.sum() < 3:
            return float("nan")
        return float(np.polyfit(times[reached], distances[inside][reached], 1)[0])

    return {
        "head_m_s": speed_along(downwind),
        "back_m_s": speed_along(-downwind),
        "flank_m_s": 0.5 * (speed_along(crosswind) + speed_along(-crosswind)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="omit for a fresh random choice")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1, 10_000))
    inputs = random_inputs(seed)
    chid = f"demo_seed{seed}"
    run_dir = RUNS_DIR / chid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{chid}.fds").write_text(build_input(inputs, chid))
    with (run_dir / "run.log").open("w") as log:
        subprocess.run(
            ["zsh", "-c", f"source ~/FDS/FDS6/bin/fds_env.zsh && fds {chid}.fds"],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    nodes, arrival = arrival_grid(run_dir)
    np.savez_compressed(run_dir / "arrival.npz", nodes_m=nodes, arrival_s=arrival)
    predicted = predicted_speeds(ros_no_wind_from_log(run_dir / f"{chid}.out"))
    measured = measured_speeds(inputs, nodes, arrival)
    summary = {"inputs": asdict(inputs), "predicted": predicted, "measured": measured}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
