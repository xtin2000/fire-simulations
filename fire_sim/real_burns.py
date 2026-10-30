"""Turn the real RxCADRE 2012 burns L1G, L2G and L2F into test samples like the training data.

Each ``real_burns/<site>.npz`` uses the training-set layout (8 m cells, row = y north, col = x
east, times in seconds after the first ignition) plus fields that say how much of it was observed:

- ``arrival_s``           first heat-camera pass that saw each cell hot (gaps filled, see below)
- ``arrival_observed``    True where the time is a real detection, False where it was gap-filled
- ``ignition_s``          earliest time a drip-torch ATV passed the cell (NaN = not lit by hand)
- ``ignition_estimated``  True where an untimed ATV route was timed from the heat camera instead
- ``fuel_depth``, ``fuel_load``, ``moisture``   measured block averages (no per-cell fuel maps)
- ``wind_times_s``, ``wind_speed_m_s``, ``wind_from_deg``   anemometer average every 120 s
- ``pass_times_s``        when the plane imaged the block; observations exist only at these times
- ``origin_utm``          UTM 16N of the grid's lower-left corner; ``utc_origin_s`` is time zero
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.path import Path as PolygonPath
from scipy.ndimage import distance_transform_edt

from fire_sim.ignition import IGNITION_GDB, atv_tracks, untimed_track_layers
from fire_sim.paths import PROJECT_ROOT, RAW_DIR
from fire_sim.wasp import arrival_map, burn_block, load_mosaics

REAL_BURNS_DIR = PROJECT_ROOT / "real_burns"
CELL_SIZE_M = 8.0
MARGIN_M = 96.0
FIRE_THRESHOLD_W_M2 = 1000.0
MAX_FILL_DISTANCE_M = 24.0
WIND_STEP_S = 120.0
MPH_TO_M_S = 0.44704


@dataclass(frozen=True)
class BurnFacts:
    """Measured, block-average inputs for one burn (RDS-2014-0028), plus the fuel model used."""

    site: str
    date_utc: str
    fuel_type: str
    fuel_model_index: int
    fuel_load_kg_m2: float
    dead_moisture: float
    live_moisture: float
    fuel_depth_m: float
    fuel_depth_source: str


BURNS = {
    "L1G": BurnFacts(
        "L1G",
        "11/04/2012",
        "grass",
        1,
        0.208,
        0.107,
        0.934,
        0.37,
        "S5 calibration (not tuned on L1G)",
    ),
    "L2G": BurnFacts(
        "L2G",
        "11/10/2012",
        "grass",
        1,
        0.267,
        0.085,
        0.822,
        0.37,
        "S5 calibration (not tuned on L2G)",
    ),
    "L2F": BurnFacts(
        "L2F",
        "11/11/2012",
        "litter",
        9,
        0.964,
        0.114,
        0.927,
        0.061,
        "FDS fuel model 9 default (no forest calibration plot)",
    ),
}


@dataclass
class Grid:
    """An 8 m grid covering a burn block plus a margin; origin is the lower-left corner (UTM)."""

    origin_x: float
    origin_y: float
    rows: int
    cols: int

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.origin_x + (np.arange(self.cols) + 0.5) * CELL_SIZE_M
        y = self.origin_y + (np.arange(self.rows) + 0.5) * CELL_SIZE_M
        return np.meshgrid(x, y)

    def cell_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        col = np.floor((np.asarray(x) - self.origin_x) / CELL_SIZE_M).astype(int)
        row = np.floor((np.asarray(y) - self.origin_y) / CELL_SIZE_M).astype(int)
        inside = (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)
        return row, col, inside


def grid_for(block_xy: np.ndarray) -> Grid:
    left, bottom = block_xy.min(axis=0) - MARGIN_M
    right, top = block_xy.max(axis=0) + MARGIN_M
    left, bottom = (
        np.floor(left / CELL_SIZE_M) * CELL_SIZE_M,
        np.floor(bottom / CELL_SIZE_M) * CELL_SIZE_M,
    )
    return Grid(
        left,
        bottom,
        int(np.ceil((top - bottom) / CELL_SIZE_M)),
        int(np.ceil((right - left) / CELL_SIZE_M)),
    )


def block_mask(grid: Grid, block_xy: np.ndarray) -> np.ndarray:
    xx, yy = grid.centres()
    return PolygonPath(block_xy).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


def observed_arrival(site: str, grid: Grid) -> np.ndarray:
    """Earliest heat-camera detection (UTC s) among the fine pixels inside each 8 m cell."""
    canvas, fine_arrival = arrival_map(site, fire_threshold_w_m2=FIRE_THRESHOLD_W_M2)
    rows, cols = fine_arrival.shape
    x = canvas.origin_xy[0] + np.arange(cols) * canvas.pixel_size_m
    y = canvas.origin_xy[1] - np.arange(rows) * canvas.pixel_size_m
    xx, yy = np.meshgrid(x, y)
    detected = np.isfinite(fine_arrival)
    row, col, inside = grid.cell_of(xx[detected], yy[detected])
    times = fine_arrival[detected][inside]
    arrival = np.full((grid.rows, grid.cols), np.inf)
    np.minimum.at(arrival, (row[inside], col[inside]), times)
    arrival[np.isinf(arrival)] = np.nan
    return arrival


def fill_gaps(arrival: np.ndarray, inside_block: np.ndarray) -> np.ndarray:
    """Give undetected block cells the time of the nearest detected cell, if one is within 24 m.

    A grass front often crosses a pixel between two passes ~4 min apart, so it is never seen hot.
    Cells farther than 24 m from any detection stay NaN (unknown, possibly unburned).
    """
    detected = np.isfinite(arrival)
    distance, (near_row, near_col) = distance_transform_edt(~detected, return_indices=True)
    filled = arrival.copy()
    fill = inside_block & ~detected & (distance * CELL_SIZE_M <= MAX_FILL_DISTANCE_M)
    filled[fill] = arrival[near_row[fill], near_col[fill]]
    return filled


def ignition_raster(
    site: str, grid: Grid, inside_block: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Earliest ATV time per cell from timed GPS; untimed routes use the heat-camera time."""
    ignition = np.full((grid.rows, grid.cols), np.inf)
    tracks = atv_tracks(site)
    row, col, inside = grid.cell_of(tracks.easting.to_numpy(), tracks.northing.to_numpy())
    keep = inside.copy()
    keep[inside] = inside_block[row[inside], col[inside]]
    np.minimum.at(ignition, (row[keep], col[keep]), tracks.time_utc_s.to_numpy()[keep])
    ignition[np.isinf(ignition)] = np.nan

    # Routes driven along the block edge (perimeter lighting) light the fuel just inside it,
    # so each route point is moved to the nearest block cell within MAX_FILL_DISTANCE_M.
    distance_out, (near_row, near_col) = distance_transform_edt(~inside_block, return_indices=True)
    estimated = np.zeros_like(inside_block)
    for layer in untimed_track_layers(site):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, _, geometry, _ = pyogrio.raw.read(IGNITION_GDB, layer=layer)
        line = shapely.from_wkb(geometry[0])
        points = shapely.line_interpolate_point(line, np.arange(0.0, line.length, CELL_SIZE_M / 2))
        coords = shapely.get_coordinates(points)
        r, c, ok = grid.cell_of(coords[:, 0], coords[:, 1])
        r, c = r[ok], c[ok]
        close = distance_out[r, c] * CELL_SIZE_M <= MAX_FILL_DISTANCE_M
        on_route = np.zeros_like(inside_block)
        on_route[near_row[r[close], c[close]], near_col[r[close], c[close]]] = True
        on_route &= inside_block & np.isnan(ignition) & np.isfinite(observed)
        ignition[on_route] = observed[on_route]
        estimated |= on_route
    return ignition, estimated


def block_wind(
    site: str, date_utc: str, start_s: float, end_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anemometer average every 120 s: (times UTC s, speed m/s, from-direction deg)."""
    columns = {
        "_Date": "date",
        "_Time_GMT_00_00": "time",
        "_WindSpeed_mph": "ws_mph",
        "_WindDirection_Degrees_from_nort": "wd",
    }
    wind = pd.read_csv(
        RAW_DIR / f"fire_behavior/Anemometer/Anemometer_Final_v2_only_{site}.csv",
        encoding="latin-1",
        usecols=list(columns),
        low_memory=False,
    ).rename(columns=columns)
    wind = wind[(wind.date.str.strip() == date_utc) & (wind.ws_mph >= 0) & (wind.wd >= 0)]
    seconds = pd.to_timedelta(wind.time.str.strip()).dt.total_seconds()
    speed = wind.ws_mph * MPH_TO_M_S
    from_rad = np.radians(wind.wd)
    frame = pd.DataFrame(
        {
            "bin": ((seconds - start_s) // WIND_STEP_S).astype(int),
            "speed": speed,
            "east": -speed * np.sin(from_rad),
            "north": -speed * np.cos(from_rad),
        }
    )
    frame = frame[(frame.bin >= 0) & (frame.bin <= int((end_s - start_s) // WIND_STEP_S))]
    means = frame.groupby("bin")[["speed", "east", "north"]].mean()
    times = start_s + means.index.to_numpy() * WIND_STEP_S
    direction = np.degrees(np.arctan2(-means.east, -means.north)) % 360.0
    return times, means.speed.to_numpy(), direction.to_numpy()


def build(site: str) -> Path:
    facts = BURNS[site]
    block_xy = burn_block(site)
    grid = grid_for(block_xy)
    inside = block_mask(grid, block_xy)

    observed = observed_arrival(site, grid)
    ignition_utc, ignition_estimated = ignition_raster(site, grid, inside, observed)
    t0 = float(np.nanmin(ignition_utc))
    arrival = fill_gaps(observed, inside)
    pass_times = np.array([m.time_utc_s for m in load_mosaics(site)])
    wind_t, wind_speed, wind_dir = block_wind(site, facts.date_utc, t0, float(pass_times.max()))

    REAL_BURNS_DIR.mkdir(exist_ok=True)
    out = REAL_BURNS_DIR / f"{site}.npz"
    np.savez_compressed(
        out,
        arrival_s=(arrival - t0).astype(np.float32),
        arrival_observed=np.isfinite(observed),
        ignition_s=(ignition_utc - t0).astype(np.float32),
        ignition_estimated=ignition_estimated,
        fuel_depth=np.where(inside, facts.fuel_depth_m, 0.0).astype(np.float32),
        fuel_load=np.where(inside, facts.fuel_load_kg_m2, 0.0).astype(np.float32),
        moisture=np.where(inside, facts.dead_moisture, 0.0).astype(np.float32),
        block_mask=inside,
        wind_times_s=wind_t - t0,
        wind_speed_m_s=wind_speed,
        wind_from_deg=wind_dir,
        pass_times_s=pass_times - t0,
        cell_size_m=np.float32(CELL_SIZE_M),
        origin_utm=np.array([grid.origin_x, grid.origin_y]),
        utc_origin_s=np.float64(t0),
        facts_json=np.array(json.dumps(asdict(facts))),
    )
    return out


def describe(path: Path) -> str:
    with np.load(path) as d:
        inside = d["block_mask"]
        arrival, observed = d["arrival_s"], d["arrival_observed"]
        lit, wind = np.isfinite(d["ignition_s"]), d["wind_speed_m_s"]
        return (
            f"{path.stem}: grid {inside.shape[0]}x{inside.shape[1]} cells, "
            f"block observed {np.mean(observed[inside]):.0%} + gap-filled "
            f"{np.mean(np.isfinite(arrival[inside]) & ~observed[inside]):.0%}, "
            f"unknown {np.mean(~np.isfinite(arrival[inside])):.0%}; "
            f"hand-lit cells {lit.sum()} ({d['ignition_estimated'].sum()} estimated); "
            f"{len(d['pass_times_s'])} passes over {d['pass_times_s'].max() / 60:.0f} min; "
            f"wind {np.mean(wind):.1f} m/s (range {wind.min():.1f}-{wind.max():.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sites", nargs="*", default=list(BURNS))
    args = parser.parse_args()
    for site in args.sites:
        print(describe(build(site)))


if __name__ == "__main__":
    main()
