"""RxCADRE 2012 ignition tracks: 1 Hz GPS of the drip-torch ATVs (RDS-2017-0065)."""

from __future__ import annotations

import warnings

import pandas as pd
import pyogrio

from fire_sim.paths import RAW_DIR

IGNITION_GDB = RAW_DIR / "ignition/RxCADRE_2011_2012_ignition_data.gdb"
SITE_LAYER_TAG = {"L1G": "B70CL1G", "L2G": "B70EL2G", "L2F": "B70EL2F"}


def atv_tracks(site: str) -> pd.DataFrame:
    """Timed ATV positions for a 2012 burn: atv, easting, northing, time_utc_s."""
    tag = SITE_LAYER_TAG[site]
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        layers = [name for name, _ in pyogrio.list_layers(IGNITION_GDB)]
        for layer in layers:
            if not layer.startswith(f"ATV_{tag}_ignition") or layer.endswith("_line"):
                continue
            meta, _, _, values = pyogrio.raw.read(IGNITION_GDB, layer=layer, read_geometry=False)
            record = dict(zip(meta["fields"], values, strict=True))
            if "UTCTIME" not in record:
                continue
            clock = pd.Series(record["UTCTIME"]).astype(str).str.zfill(6)
            seconds = (
                clock.str[:2].astype(int) * 3600
                + clock.str[2:4].astype(int) * 60
                + clock.str[4:].astype(int)
            )
            frames.append(
                pd.DataFrame(
                    {
                        "atv": layer.removeprefix(f"ATV_{tag}_"),
                        "easting": record["EASTING"],
                        "northing": record["NORTHING"],
                        "time_utc_s": seconds.to_numpy(dtype=float),
                        "speed_m_s": record["SOG"],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True).sort_values(["atv", "time_utc_s"])


def untimed_track_layers(site: str) -> list[str]:
    """Ignition routes that exist only as lines, without a usable timestamp per point."""
    tag = SITE_LAYER_TAG[site]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        layers = [name for name, _ in pyogrio.list_layers(IGNITION_GDB)]
    timed = set(atv_tracks(site).atv)
    return [
        layer
        for layer in layers
        if layer.startswith(f"ATV_{tag}_ignition")
        and layer.endswith("_line")
        and layer.removeprefix(f"ATV_{tag}_").removesuffix("_line") not in timed
    ]
