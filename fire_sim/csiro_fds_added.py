"""The six added CSIRO fires in FDS's full physics mode, built from NIST's C064 and F19 inputs.

NIST set up only C064 and F19. For C062, C112, E11, F04, F05 and F16 this module builds the same
kind of input from the same two templates, with what the field crews measured and nothing tuned:

  physics   the template of the fire's grass: C064's for Eriachne (ER), F19's for Themeda (TH).
            That gives the grass chemistry, the ignitor's strength and burn time, and the
            blade sampling NIST chose for that grass
  domain    the 1 m grids of NIST's two fires: C064's 120 m square for the 100 m plots, F19's
            240 m square for the 190 to 210 m plots
  plot      the scanned plot size and the 16-sample patchwork, as in ELMFIRE
  lighting  two walkers from the line's centre at 1.3 m/s, on the line's centre and tilt (a tilted
            line is a staircase of strips that all light from the line's centre)
  wind      the two upwind corners every 5 s, turning with the weather station's direction
  weather   the fire's air temperature and dead-grass moisture, and the run length of ELMFIRE

The surrounding dirt roads of NIST's inputs are left out: they bound the plots NIST scanned, and
there is no grass outside the plot anyway.

Commands:
  setup  write runs/csiro_full/<case>/<case>.fds for the six fires
  run    run every unfinished case, one after another
"""

from __future__ import annotations

import argparse
import math
import re

from fire_sim import annaburroo
from fire_sim import csiro_fds_full as full
from fire_sim import elmfire_csiro as elm
from fire_sim.csiro_fds_full import FullCase, Grid
from fire_sim.elmfire_csiro import Conditions, CsiroFire

NIST_SMALL, NIST_LARGE = elm.FIRES
GRASS_TEMPLATE = {"ER": NIST_SMALL, "TH": NIST_LARGE}
LARGE_PLOT_M = 150.0
LINE_STEP_M = 1.0

ROAD_VENTS = re.compile(r"&VENT [^\n]*SURF_ID='ROAD' /\n")
SPREADING_VENT = re.compile(r"&VENT XB=0\.0,([\d.]+),[^/]*SURF_ID='IGNITION'[^/]*/")
DEVC_LINES = re.compile(r"&DEVC ID='(?:x|D)'[^\n]*\n?")


def added_cases() -> tuple[FullCase, ...]:
    """One 1 m run per added fire, with the station's wind direction."""
    return tuple(
        FullCase(fire, Grid.ONE_METRE, Conditions.FIELD_STATION, turns_with_station=True)
        for fire in elm.all_fires()[len(elm.FIRES) :]
    )


def _template(fire: CsiroFire) -> str:
    return (full.FULL_INPUTS_DIR / f"{full.nist_name(fire, Grid.ONE_METRE)}.fds").read_text()


def grass_template_fire(fire: CsiroFire) -> CsiroFire:
    return GRASS_TEMPLATE[annaburroo.plot_average(fire.name)["FuelType"]]


def layout_template_fire(fire: CsiroFire) -> CsiroFire:
    return NIST_SMALL if fire.plot_length_m <= LARGE_PLOT_M else NIST_LARGE


def with_domain(text: str, layout_text: str) -> str:
    """Swap in the meshes and the head sensors of the template that has the right domain."""
    meshes = full.MESH_AND_MULT.search(layout_text).group(0)
    text = full.MESH_AND_MULT.sub(lambda _: meshes, text)
    sensors = "".join(DEVC_LINES.findall(layout_text))
    stripped = DEVC_LINES.sub("", text)
    return stripped.replace("&TAIL /", f"{sensors}\n&TAIL /")


def lighting_strips(fire: CsiroFire, width_m: float) -> list[tuple[float, float, float, float]]:
    """(x0, x1, y0, y1) strips that make up the lighting line, centred on its middle."""
    x0 = max(fire.lighting_x_m, 0.0)
    half = fire.field_lighting_half_length_m
    centre_y = fire.lighting_centre_y_m
    if fire.lighting_tilt_deg == 0:
        return [(x0, x0 + width_m, centre_y - half, centre_y + half)]
    tilt = math.radians(fire.lighting_tilt_deg)
    steps = int(2 * half / LINE_STEP_M) + 1
    strips = []
    for n in range(steps):
        along = -half + n * LINE_STEP_M
        x = x0 + along * math.sin(tilt)
        y = centre_y + along * math.cos(tilt)
        strips.append((x, x + width_m, y - LINE_STEP_M / 2, y + LINE_STEP_M / 2))
    return strips


def added_lighting(text: str, fire: CsiroFire) -> str:
    """NIST's ignitor, lit outward from the line's centre at the walkers' pace and reach."""
    width = float(SPREADING_VENT.search(text).group(1))
    centre = f"{max(fire.lighting_x_m, 0.0):g},{fire.lighting_centre_y_m:g},0"
    vents = [
        f"&VENT XB={x0:.2f},{x1:.2f},{y0:.2f},{y1:.2f},0.,0., SURF_ID='IGNITION', XYZ={centre}, "
        f"SPREAD_RATE={fire.field_walking_speed_m_s:.3f} /"
        for x0, x1, y0, y1 in lighting_strips(fire, width)
    ]
    return SPREADING_VENT.sub(lambda _: "\n".join(vents), text)


def fire_weather(text: str, fire: CsiroFire) -> str:
    """This fire's air temperature, grass moisture and run length in place of the template's."""
    temperature = annaburroo.plot_average(fire.name)["Temp.C"]
    text = re.sub(r"TMPA=[\d.]+", f"TMPA={temperature:g}", text)
    text = re.sub(
        r"MOISTURE_CONTENT\s*=\s*[\d.]+",
        f"MOISTURE_CONTENT         = {fire.dead_moisture_percent / 100:.3f}",
        text,
    )
    return re.sub(r"T_END=[\d.]+", f"T_END={fire.end_time_s:g}", text)


def added_input(case: FullCase) -> str:
    """The full-physics input for one added fire."""
    fire = case.fire
    text = ROAD_VENTS.sub("", _template(grass_template_fire(fire)))
    text = with_domain(text, _template(layout_template_fire(fire)))
    text = fire_weather(text, fire)
    text = re.sub(r"TITLE='[^']*'", f"TITLE='CSIRO grassland fire, case {fire.name}'", text)
    return full.field_input(text, case, added_lighting)


def setup() -> None:
    for case in added_cases():
        print(full.write_case(case, added_input(case)))


def run() -> None:
    full.run_cases(added_cases())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["setup", "run"])
    {"setup": setup, "run": run}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
