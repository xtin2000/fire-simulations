"""CSIRO's 1986 Annaburroo field data: fuel samples, plot geometry and corner wind.

Data: Gould, Cruz & Sullivan (2024), 1986 Annaburroo Experimental Grassland Fire Data, CSIRO,
doi:10.25919/ycd3-w209, CC BY 4.0.

Each plot has a hand-drawn fuel map with 16 samples on a 4 x 4 grid (load in t/ha over height in
m, grass E = Eriachne or T = Themeda). They are typed into data/derived/csiro_fuel_samples.csv
in simulation coordinates: x downwind from the lighting edge, y across the plot, positive to the
left of the fire's travel. Sample positions are read off the scans to about +-5 m.

Fires C062, C112, E11, F04, F05 and F16 have no NIST input files, so their plot size and lighting
line are read off the same scans, to about +-5 m, into data/derived/csiro_plot_geometry.csv. Only
the line's position and tilt come from the scan: its length is CSIRO's table value, because the
fire edges drawn on the scan have already spread sideways. All six were lit on the east side and
the weather station puts the wind behind that side, so they run in the same frame as F19. On E11
one load is struck out on the sheet (15 samples); on F05 one height is missing, filled with the
mean of the plot's samples of the same grass.

The wind file gives the 2 m wind every 5 s at the four plot corners. The fire runs from the
upwind edge, so only the two upwind corners see wind the fire has not disturbed. For F19 their
average matches Mell et al. (2007), Fig. 13a, to 0.06 m/s, and CSIRO's burn table, the site
weather station and the scan agree the fire ran west. For C064 the sources disagree: the scan's
north arrow puts the lit edge on the south side (upwind SE and SW, used here; closest to Mell et
al., Fig. 14a), but the burn table gives a head bearing of 280 degrees and the weather station a
wind from the east, which would make NE and SE upwind (4.4 instead of 4.8 m/s during the fire).

The corner anemometers record speed only. Wind direction is in AWS_Data.csv at the top of the
dataset: one site weather station, 10 m high, one reading per minute.

The measured fire-head positions and outlines over time come from NIST's validation data
(firemodels/exp), digitised from CSIRO's photos; every engine is scored against them.

Command:
  download  fetch the dataset (584 MB) and unpack the fire folders and site tables, and fetch
            NIST's measured fire-head positions for C064 and F19 into data/raw/csiro/measured/
"""

from __future__ import annotations

import argparse
import math
import subprocess
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fire_sim.paths import CSIRO_RAW_DIR, DERIVED_DIR

DATASET_URL = "https://data.csiro.au/dap/ws/v2/collections/60062/data/70770839"
MEASURED_URL = "https://raw.githubusercontent.com/firemodels/exp/master/CSIRO_Grassland_Fires"
ANNABURROO_DIR = CSIRO_RAW_DIR / "annaburroo"
FUEL_SAMPLES_CSV = DERIVED_DIR / "csiro_fuel_samples.csv"
PLOT_GEOMETRY_CSV = DERIVED_DIR / "csiro_plot_geometry.csv"
FIRE_NAMES = ("C064", "F19")
ADDED_FIRE_NAMES = ("C062", "C112", "E11", "F04", "F05", "F16")
T_HA_TO_KG_M2 = 0.1
# Mean blade surface-to-volume ratio of each grass (Mell et al. 2007).
GRASS_SURFACE_TO_VOLUME_PER_M = {"E": 9770, "T": 12240}


@dataclass(frozen=True)
class UpwindCorners:
    """The upwind corners, named as in the wind file, on each side of the fire's travel."""

    right: str
    left: str


# CSIRO's burn table (head bearing 280 degrees) and the weather station (wind from the east) put
# the upwind edge of both fires on the east side.
EAST_CORNERS = UpwindCorners("NE", "SE")
UPWIND_CORNERS = {
    "C064": UpwindCorners("SE", "SW"),
    "F19": EAST_CORNERS,
    **dict.fromkeys(ADDED_FIRE_NAMES, EAST_CORNERS),
}


@dataclass(frozen=True)
class PlotGeometry:
    """A plot and its lighting line, in metres: x west of the east edge, y south of the centre."""

    east_west_m: float
    north_south_m: float
    lighting_x_m: float
    lighting_y_m: float
    lighting_length_m: float
    lighting_tilt_deg: float

    def lighting_ends(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The line's (x, y) ends, north end first. A positive tilt moves x west going south."""
        half = self.lighting_length_m / 2
        along_x = half * math.sin(math.radians(self.lighting_tilt_deg))
        along_y = half * math.cos(math.radians(self.lighting_tilt_deg))
        return (
            (self.lighting_x_m - along_x, self.lighting_y_m - along_y),
            (self.lighting_x_m + along_x, self.lighting_y_m + along_y),
        )


def plot_geometry(fire_name: str) -> PlotGeometry:
    """The plot and lighting line read off a fire's scanned sheet."""
    row = pd.read_csv(PLOT_GEOMETRY_CSV).set_index("fire").loc[fire_name]
    return PlotGeometry(
        east_west_m=row.plot_east_west_m,
        north_south_m=row.plot_north_south_m,
        lighting_x_m=row.lighting_x_m,
        lighting_y_m=row.lighting_y_m,
        lighting_length_m=row.lighting_length_m,
        lighting_tilt_deg=row.lighting_tilt_deg,
    )


def head_speed(times_s: np.ndarray, positions_m: np.ndarray) -> float:
    """Average speed between the first and last observation."""
    return float((positions_m[-1] - positions_m[0]) / (times_s[-1] - times_s[0]))


def download() -> None:
    """Fetch the dataset zip, unpack the fire folders and fetch the measured fronts."""
    archive = ANNABURROO_DIR / "dataset.zip"
    if not archive.exists():
        ANNABURROO_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", "-o", str(archive), DATASET_URL], check=True)
    with zipfile.ZipFile(archive) as dataset:
        unpacked = (*FIRE_NAMES, *ADDED_FIRE_NAMES)
        site_files = ("Site and block maps", "ReadMe - datafile descriptions.txt")
        members = [m for m in dataset.namelist() if m.split("/")[0] in (*unpacked, *site_files)]
        site_tables = ["AWS_Data.csv", "Burn_Period_Data.csv", "Plot_Average_Data.csv"]
        dataset.extractall(ANNABURROO_DIR, members=members + site_tables)
    print(
        f"unpacked {', '.join(unpacked)}, the site tables, the site maps and the ReadMe "
        f"into {ANNABURROO_DIR}"
    )
    measured_dir = CSIRO_RAW_DIR / "measured"
    measured_dir.mkdir(parents=True, exist_ok=True)
    for fire_name in FIRE_NAMES:
        for kind in ("exp", "flame_position_data", "head_depth"):
            name = f"Case_{fire_name}_{kind}.csv"
            subprocess.run(
                ["curl", "-sfL", "-o", str(measured_dir / name), f"{MEASURED_URL}/{name}"],
                check=True,
            )
    print(f"fetched the measured fire-head positions into {measured_dir}")


def _clock_seconds(hhmmss: pd.Series) -> pd.Series:
    text = hhmmss.astype(str).str.zfill(6)
    hours, minutes, seconds = (text.str[i : i + 2].astype(int) for i in (0, 2, 4))
    return hours * 3600 + minutes * 60 + seconds


def lighting_clock_seconds(fire_name: str) -> int:
    """The clock time the fire was lit, from CSIRO's burn table, in seconds after midnight."""
    table = pd.read_csv(ANNABURROO_DIR / "Burn_Period_Data.csv")
    lit = table.loc[table.FireID == fire_name, "IgnTime.hhmmss"].iloc[:1]
    return int(_clock_seconds(lit).iloc[0])


def upwind_wind(fire_name: str, *, corners: UpwindCorners | None = None) -> pd.DataFrame:
    """2 m wind (m/s) at the right and left upwind corners, by seconds after lighting.

    The corners default to those read off the fire's fuel sheet.
    """
    path = ANNABURROO_DIR / fire_name / f"{fire_name}_2mwind.csv"
    wind = pd.read_csv(path, encoding="utf-8-sig")
    wind.columns = wind.columns.str.strip()
    corners = corners or UPWIND_CORNERS[fire_name]
    return pd.DataFrame(
        {
            "time_s": _clock_seconds(wind["Time.hhmmss"]) - lighting_clock_seconds(fire_name),
            "right_m_s": wind[f"{corners.right}2mWind.ms"],
            "left_m_s": wind[f"{corners.left}2mWind.ms"],
        }
    )


def burn_periods(fire_name: str) -> pd.DataFrame:
    """CSIRO's head-fire record: distance from lighting (m) and bearing at each mapped time (s)."""
    table = pd.read_csv(ANNABURROO_DIR / "Burn_Period_Data.csv")
    periods = table[table.FireID == fire_name]
    return pd.DataFrame(
        {
            "time_s": periods["TimeSinceIgn"].to_numpy(),
            "distance_m": periods["CumDist.m"].to_numpy(),
            "bearing_deg": periods["FireDir.deg"].to_numpy(),
        }
    )


def weather_station_wind(fire_name: str) -> pd.DataFrame:
    """The site weather station's 10 m wind on the fire's day, by seconds after lighting.

    direction_deg is the compass bearing the wind blows from, one reading per minute.
    """
    table = pd.read_csv(ANNABURROO_DIR / "Burn_Period_Data.csv")
    burn_date = table.loc[table.FireID == fire_name, "Date.yymmdd"].iloc[0]
    station = pd.read_csv(ANNABURROO_DIR / "AWS_Data.csv")
    day = station[station["AWSDate.yymmdd"] == burn_date]
    return pd.DataFrame(
        {
            "time_s": (
                _clock_seconds(day["AWSTime.hhmmss"]) - lighting_clock_seconds(fire_name)
            ).to_numpy(),
            "speed_10m_m_s": day["10mAWSWind.ms"].to_numpy(),
            "direction_deg": day["10mAWSDir.deg"].to_numpy(),
        }
    )


def plot_average(fire_name: str) -> pd.Series:
    """CSIRO's plot-average row for a fire: wind, moisture, grass type, load and height."""
    table = pd.read_csv(ANNABURROO_DIR / "Plot_Average_Data.csv", encoding="latin1")
    return table.set_index("FireID").loc[fire_name]


def fuel_samples(fire_name: str) -> pd.DataFrame:
    """The fire's fuel samples with load in kg/m2 and the grass's blade fineness added."""
    samples = pd.read_csv(FUEL_SAMPLES_CSV).query("fire == @fire_name").reset_index(drop=True)
    samples["load_kg_m2"] = samples.load_t_ha * T_HA_TO_KG_M2
    samples["surface_to_volume_per_m"] = samples.grass.map(GRASS_SURFACE_TO_VOLUME_PER_M)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("download",))
    parser.parse_args()
    download()


if __name__ == "__main__":
    main()
