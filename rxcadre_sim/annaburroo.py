"""CSIRO's 1986 Annaburroo field data for grass fires C064 and F19: fuel samples and corner wind.

Data: Gould, Cruz & Sullivan (2024), 1986 Annaburroo Experimental Grassland Fire Data, CSIRO,
doi:10.25919/ycd3-w209, CC BY 4.0.

Each plot has a hand-drawn fuel map with 16 samples on a 4 x 4 grid (load in t/ha over height in
m, grass E = Eriachne or T = Themeda). They are typed into data/derived/csiro_fuel_samples.csv
in simulation coordinates: x downwind from the lighting edge, y across the plot, positive to the
left of the fire's travel. Sample positions are read off the scans to about +-5 m.

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
  download  fetch the dataset (584 MB) and unpack the C064 and F19 folders and site tables, and
            fetch NIST's measured fire-head positions into data/raw/csiro/measured/
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rxcadre_sim.paths import CSIRO_RAW_DIR, DERIVED_DIR

DATASET_URL = "https://data.csiro.au/dap/ws/v2/collections/60062/data/70770839"
MEASURED_URL = "https://raw.githubusercontent.com/firemodels/exp/master/CSIRO_Grassland_Fires"
ANNABURROO_DIR = CSIRO_RAW_DIR / "annaburroo"
FUEL_SAMPLES_CSV = DERIVED_DIR / "csiro_fuel_samples.csv"
FIRE_NAMES = ("C064", "F19")
# Both fires were lit at 12:56:00: every photo time minus its measured fire time gives it.
LIGHTING_CLOCK_S = 12 * 3600 + 56 * 60
T_HA_TO_KG_M2 = 0.1
# Mean blade surface-to-volume ratio of each grass (Mell et al. 2007).
GRASS_SURFACE_TO_VOLUME_PER_M = {"E": 9770, "T": 12240}


@dataclass(frozen=True)
class UpwindCorners:
    """The upwind corners, named as in the wind file, on each side of the fire's travel."""

    right: str
    left: str


UPWIND_CORNERS = {"C064": UpwindCorners("SE", "SW"), "F19": UpwindCorners("NE", "SE")}
# CSIRO's burn table (head bearing 280 degrees) and the weather station (wind from the east) put
# the upwind edge of both fires on the east side.
EAST_CORNERS = UpwindCorners("NE", "SE")


def head_speed(times_s: np.ndarray, positions_m: np.ndarray) -> float:
    """Average speed between the first and last observation."""
    return float((positions_m[-1] - positions_m[0]) / (times_s[-1] - times_s[0]))


def download() -> None:
    """Fetch the dataset zip, unpack the folders of the two fires and fetch the measured fronts."""
    archive = ANNABURROO_DIR / "dataset.zip"
    if not archive.exists():
        ANNABURROO_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", "-o", str(archive), DATASET_URL], check=True)
    with zipfile.ZipFile(archive) as dataset:
        members = [m for m in dataset.namelist() if m.split("/")[0] in FIRE_NAMES]
        site_tables = ["AWS_Data.csv", "Burn_Period_Data.csv", "Plot_Average_Data.csv"]
        dataset.extractall(ANNABURROO_DIR, members=members + site_tables)
    print(f"unpacked {', '.join(FIRE_NAMES)} and the site tables into {ANNABURROO_DIR}")
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
            "time_s": _clock_seconds(wind["Time.hhmmss"]) - LIGHTING_CLOCK_S,
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
            "time_s": (_clock_seconds(day["AWSTime.hhmmss"]) - LIGHTING_CLOCK_S).to_numpy(),
            "speed_10m_m_s": day["10mAWSWind.ms"].to_numpy(),
            "direction_deg": day["10mAWSDir.deg"].to_numpy(),
        }
    )


def fuel_samples(fire_name: str) -> pd.DataFrame:
    """The 16 fuel samples with load in kg/m2 and the grass's blade fineness added."""
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
