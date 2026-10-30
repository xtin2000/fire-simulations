"""Download the public RxCADRE 2012 datasets this project uses into data/raw/.

All come from the USDA Forest Service Research Data Archive. Only the S5
files are pulled out of the 1.5 GB fire-behavior archive (range requests).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from fire_sim.paths import RAW_DIR
from fire_sim.remote_zip import extract_members

ARCHIVE = "https://www.fs.usda.gov/rds/archive/products"


@dataclass(frozen=True)
class Dataset:
    """One archive (or the needed slice of it) and where it lands under data/raw/."""

    name: str
    url: str
    patterns: tuple[str, ...]
    strip_prefix: str
    citation: str


DATASETS = (
    Dataset(
        name="fuels",
        url=f"{ARCHIVE}/RDS-2014-0028/RDS-2014-0028_Data.zip",
        patterns=("Data/",),
        strip_prefix="Data/",
        citation="Ottmar & Restaino 2014, https://doi.org/10.2737/RDS-2014-0028",
    ),
    Dataset(
        name="fire_behavior",
        url=f"{ARCHIVE}/RDS-2016-0038/RDS-2016-0038_Data_1_of_3.zip",
        patterns=(
            "FireBehavior/s5/",
            "Anemometer_Final_v2_only_Splots.csv",
            "Anemometer_Final_v2_only_L1G.csv",
            "Anemometer_Final_v2_only_L2G.csv",
            "Anemometer_Final_v2_only_L2F.csv",
        ),
        strip_prefix="Data/",
        citation="Jimenez & Butler 2016, https://doi.org/10.2737/RDS-2016-0038",
    ),
    Dataset(
        name="supplements",
        url=f"{ARCHIVE}/RDS-2016-0038/RDS-2016-0038_Supplements.zip",
        patterns=("Supplements/",),
        strip_prefix="Supplements/",
        citation="Jimenez & Butler 2016, https://doi.org/10.2737/RDS-2016-0038",
    ),
    Dataset(
        name="survey",
        url=f"{ARCHIVE}/RDS-2016-0014/RDS-2016-0014.zip",
        patterns=("Data/",),
        strip_prefix="Data/",
        citation="RxCADRE 2012 instrument survey locations, https://doi.org/10.2737/RDS-2016-0014",
    ),
    Dataset(
        name="wasp_mosaics",
        url=f"{ARCHIVE}/RDS-2016-0008/RDS-2016-0008.zip",
        patterns=("Data/2012_L", "Supplements/"),
        strip_prefix="Data/",
        citation="Hudak et al. 2016, https://doi.org/10.2737/RDS-2016-0008",
    ),
    Dataset(
        name="wasp_frames",
        url=f"{ARCHIVE}/RDS-2016-0007/RDS-2016-0007_Data_2_of_2.zip",
        patterns=("_allimages_w_time.csv",),
        strip_prefix="Data/",
        citation="Hudak et al. 2016, https://doi.org/10.2737/RDS-2016-0007",
    ),
    Dataset(
        name="ignition",
        url=f"{ARCHIVE}/RDS-2017-0065/RDS-2017-0065.zip",
        patterns=("Data/",),
        strip_prefix="Data/",
        citation="RxCADRE 2011 and 2012 ignition data, https://doi.org/10.2737/RDS-2017-0065",
    ),
    Dataset(
        name="burn_blocks",
        url=f"{ARCHIVE}/RDS-2014-0031/RDS-2014-0031.zip",
        patterns=("Data/",),
        strip_prefix="Data/",
        citation="RxCADRE 2008-2012 burn blocks, https://doi.org/10.2737/RDS-2014-0031",
    ),
)


def download(names: list[str] | None = None) -> None:
    """Fetch the named datasets (all of them when ``names`` is empty) into data/raw/."""
    for dataset in DATASETS:
        if names and dataset.name not in names:
            continue
        written = extract_members(
            dataset.url,
            list(dataset.patterns),
            RAW_DIR / dataset.name,
            strip_prefix=dataset.strip_prefix,
        )
        print(f"{dataset.name}: {len(written)} files ({dataset.citation})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[d.name for d in DATASETS], nargs="*")
    download(parser.parse_args().only)


if __name__ == "__main__":
    main()
