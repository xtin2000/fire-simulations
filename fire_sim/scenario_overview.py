"""Plot 12 exported scenarios spanning the range of burned area, as a quick visual check."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fire_sim.scenarios import DOMAIN_SIZE_M

PANEL_COUNT = 12


def plot_overview(export_dir: Path, image_path: Path) -> Path:
    manifest = pd.read_csv(export_dir / "manifest.csv").set_index("scenario_id")
    by_area = manifest.sort_values("burned_area_m2").index.to_numpy()
    picks = by_area[np.linspace(0, len(by_area) - 1, PANEL_COUNT).astype(int)]
    extent = [0, DOMAIN_SIZE_M, 0, DOMAIN_SIZE_M]

    figure, axes_grid = plt.subplots(3, 4, figsize=(14, 10.5))
    for axes, scenario_id in zip(axes_grid.flat, picks, strict=True):
        with np.load(export_dir / f"{scenario_id}.npz") as data:
            fuel_depth, arrival_min = data["fuel_depth"], data["arrival_s"] / 60
        row = manifest.loc[scenario_id]
        axes.imshow(fuel_depth, origin="lower", extent=extent, cmap="Greens", alpha=0.5)
        image = axes.imshow(
            np.ma.masked_invalid(arrival_min),
            origin="lower",
            extent=extent,
            cmap="inferno_r",
            vmin=0,
            vmax=60,
        )
        firebreak = ", firebreak" if row.has_firebreak else ""
        axes.set_title(
            f"{scenario_id}: {row.wind_speed_m_s} m/s from {row.wind_from_deg:.0f}°, "
            f"M={row.dead_moisture}\n{row.ignition_kind}{firebreak}, "
            f"{row.burned_area_m2 / 1e4:.1f} ha",
            fontsize=8,
        )
        axes.set_xticks([])
        axes.set_yticks([])
    figure.colorbar(image, ax=axes_grid, shrink=0.6, label="fire arrival time (min)")
    figure.savefig(image_path, dpi=70)
    plt.close(figure)
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/scenario_overview.png"))
    args = parser.parse_args()
    print(plot_overview(args.export_dir, args.out))


if __name__ == "__main__":
    main()
