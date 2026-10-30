"""Read RxCADRE 2012 WASP long-wave infrared mosaics as timed, georeferenced fire observations.

Each mosaic is one aircraft pass over a burn block (RDS-2016-0008); its time is the mean
acquisition time of its frames (RDS-2016-0007 frame tables). Pixels are 3 m, UTM 16N, and
hold ground-leaving radiant exitance in W/m^2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import tifffile
from matplotlib.path import Path as PolygonPath

from fire_sim.paths import RAW_DIR

MOSAIC_DIR = RAW_DIR / "wasp_mosaics"
FRAME_TABLE_DIR = RAW_DIR / "wasp_frames"
BURN_BLOCKS = RAW_DIR / "burn_blocks/2012_fire_perimeters/2012_burn_block_perimeters.shp"
LARGE_BURNS = ("L1G", "L2G", "L2F")
NODATA = 65535


@dataclass
class Mosaic:
    """One aircraft pass: exitance raster plus where and when it was taken."""

    site: str
    first_frame: int
    last_frame: int
    time_utc_s: float
    exitance_w_m2: np.ndarray  # (rows, cols); NaN where the pass saw nothing
    origin_xy: tuple[float, float]  # UTM of the upper-left pixel centre
    pixel_size_m: float

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.exitance_w_m2.shape
        x = self.origin_xy[0] + np.arange(cols) * self.pixel_size_m
        y = self.origin_xy[1] - np.arange(rows) * self.pixel_size_m
        return x, y


def frame_times(site: str) -> pd.Series:
    """UTC seconds-of-day for every frame number at a site."""
    table = pd.read_csv(FRAME_TABLE_DIR / f"{site}_allimages_w_time.csv")
    table.columns = [c.strip() for c in table.columns]
    seconds = table.Hour * 3600 + table.Min * 60 + table.S
    return pd.Series(seconds.to_numpy(dtype=float), index=table["frame #"].to_numpy())


def burn_block(site: str) -> np.ndarray:
    reader = shapefile.Reader(str(BURN_BLOCKS))
    for record, shape in zip(reader.records(), reader.shapes(), strict=True):
        if record.as_dict()["Block_Name"].strip() == site:
            return np.array(shape.points)
    raise LookupError(site)


def load_mosaics(site: str) -> list[Mosaic]:
    times = frame_times(site)
    mosaics = []
    for tif in sorted((MOSAIC_DIR / f"2012_{site}").glob(f"{site}_Wm2_mosaic_*.tif")):
        first, last = map(int, re.search(r"_(\d+)_(\d+)\.tif$", tif.name).groups())
        world = [float(v) for v in tif.with_suffix(".tfw").read_text().split()]
        raster = tifffile.imread(tif).astype(np.float32)
        raster[(raster <= 0) | (raster >= NODATA)] = np.nan
        frames = times.loc[(times.index >= first) & (times.index <= last)]
        mosaics.append(
            Mosaic(site, first, last, float(frames.mean()), raster, (world[4], world[5]), world[0])
        )
    return sorted(mosaics, key=lambda m: m.time_utc_s)


def block_mask(mosaic: Mosaic, block_xy: np.ndarray) -> np.ndarray:
    x, y = mosaic.pixel_centres()
    xx, yy = np.meshgrid(x, y)
    inside = PolygonPath(block_xy).contains_points(np.c_[xx.ravel(), yy.ravel()])
    return inside.reshape(xx.shape)


def pass_summary(site: str, *, fire_threshold_w_m2: float) -> pd.DataFrame:
    """Per pass: time, share of the burn block imaged, and active-fire area inside the block."""
    block_xy = burn_block(site)
    rows = []
    for mosaic in load_mosaics(site):
        inside = block_mask(mosaic, block_xy)
        seen = inside & np.isfinite(mosaic.exitance_w_m2)
        fire = seen & (mosaic.exitance_w_m2 > fire_threshold_w_m2)
        rows.append(
            {
                "site": site,
                "frames": f"{mosaic.first_frame}-{mosaic.last_frame}",
                "time_utc": pd.to_timedelta(round(mosaic.time_utc_s), unit="s"),
                "block_coverage": seen.sum() / inside.sum(),
                "fire_area_ha": fire.sum() * mosaic.pixel_size_m**2 / 1e4,
                "background_w_m2": float(np.nanmedian(mosaic.exitance_w_m2[seen]))
                if seen.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def mosaic_files(site: str) -> list[Path]:
    return sorted((MOSAIC_DIR / f"2012_{site}").glob(f"{site}_Wm2_mosaic_*.tif"))


def arrival_map(site: str, *, fire_threshold_w_m2: float) -> tuple[Mosaic, np.ndarray]:
    """First pass time (UTC s) at which each pixel exceeded the threshold; NaN if never.

    Passes of one site share a pixel size and are offset by whole pixels, so each is pasted
    into a common canvas covering all passes. Returns a Mosaic describing that canvas
    (its exitance is the last pass) together with the arrival grid.
    """
    mosaics = load_mosaics(site)
    pixel = mosaics[0].pixel_size_m
    left = min(m.origin_xy[0] for m in mosaics)
    top = max(m.origin_xy[1] for m in mosaics)
    right = max(m.origin_xy[0] + (m.exitance_w_m2.shape[1] - 1) * pixel for m in mosaics)
    bottom = min(m.origin_xy[1] - (m.exitance_w_m2.shape[0] - 1) * pixel for m in mosaics)
    shape = (round((top - bottom) / pixel) + 1, round((right - left) / pixel) + 1)

    arrival = np.full(shape, np.nan)
    latest = np.full(shape, np.nan, dtype=np.float32)
    for mosaic in mosaics:
        row0 = round((top - mosaic.origin_xy[1]) / pixel)
        col0 = round((mosaic.origin_xy[0] - left) / pixel)
        rows, cols = mosaic.exitance_w_m2.shape
        window = (slice(row0, row0 + rows), slice(col0, col0 + cols))
        seen = np.isfinite(mosaic.exitance_w_m2)
        latest[window][seen] = mosaic.exitance_w_m2[seen]
        newly_hot = np.isnan(arrival[window]) & (mosaic.exitance_w_m2 > fire_threshold_w_m2)
        arrival[window][newly_hot] = mosaic.time_utc_s
    canvas = Mosaic(
        site,
        mosaics[0].first_frame,
        mosaics[-1].last_frame,
        mosaics[-1].time_utc_s,
        latest,
        (left, top),
        pixel,
    )
    return canvas, arrival
