#!/usr/bin/env python3
"""
Run the lnd_type_plus analysis for one model group, scenario, and year, and split the result
into TWO PMTiles archives -- one carrying only the water classes, one only the land classes
-- so the two can be styled and toggled independently on the map. In each archive the other
group's pixels are fully transparent.

Value -> label -> color (from the CPRA land/water condition legend):

\b
      1  Vegetated Wetland                                    #E4E4E4   land
      2  Water - Deep Subtidal      (Depths > 1.0 m)          #FFFFFF   water
      3  Bare/Unvegetated Wetland                             #DDDDDD   land
      4  Upland/Developed Areas                               #D6D6D6   land
      5  Flotant Marsh                                        #EAEAEA   land
      6  Water - Shallow Intertidal (0.0 m < Depths < 0.46 m) #F1F1F1   water
      7  Water - Shallow Subtidal   (0.46 m < Depths < 1.0 m) #F8F8F8   water
  nodata                                                      transparent

With --no-depths (the default) every water class is drawn in the deep-water white, so the
water archive is a plain water mask; --depths gives each depth class its own color as above.

The split follows the legend's own labels: the three "Water - " classes are water, the rest
are land. Flotant Marsh (5) is a vegetated mat and lands on the land side, even though it
floats.

Reads from CPRA_MP_DATA_BASE_PATH (crosswalks from CPRA_MP_DATA_CROSSWALKS_PATH), set in the
environment or .env.

Writes <output>/lnd_type_plus_mg<model group>_s<scenario>_<year>_water.pmtiles and
..._land.pmtiles.
"""

import tempfile
from pathlib import Path

import click
import numpy as np
import rasterio
import rasterio.shutil
from pmtiles.convert import mbtiles_to_pmtiles
from rasterio.enums import Resampling

from cpra.mp.data.analyses import resolve_analysis
from cpra.mp.data.analysis.execution.execute import run_to_path

# lnd_type_plus class colors with --depths. --no-depths draws every water class in
# DEEP_WATER instead.
DEEP_WATER = 2
WATER_COLORS = {
    DEEP_WATER: "#FFFFFF",
    6: "#F1F1F1",
    7: "#F8F8F8",
}
LAND_COLORS = {
    1: "#E4E4E4",
    3: "#DDDDDD",
    4: "#D6D6D6",
    5: "#EAEAEA",
}

# The scenario drawn when none is given: the 50th percentile ESLR member, which also carries
# the most weight in lnd_conf.
DEFAULT_SCENARIO_ID = 34

# Overview decimation factors, i.e. the zoom levels below the native one. Adjust if you need
# more/fewer zoom levels for this raster's extent.
OVERVIEW_FACTORS = [2, 4, 8, 16, 32, 64]


def hex_to_rgba(color: str):
    h = color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


def group_colors(depths: bool):
    """{group: {class value: color}}. Only the group's own classes are listed, which is
    what performs the split: every other value, nodata included, is left transparent."""
    water = WATER_COLORS if depths else dict.fromkeys(WATER_COLORS, WATER_COLORS[DEEP_WATER])
    return {"water": water, "land": LAND_COLORS}


def run_lnd_type_plus(model_group_id: int, scenario_id: int, year: int, path: Path):
    """Run lnd_type_plus for one (model group, scenario, year) and write it to `path`."""
    resolved = resolve_analysis(
        "lnd_type_plus",
        filters={"model_group_id": model_group_id, "scenario_id": scenario_id, "calendar_year": [year]},
    )
    if not run_to_path(resolved.pipeline, path):
        raise click.ClickException(
            f"lnd_type_plus returned no data for model_group_id={model_group_id}, scenario_id={scenario_id}, {year}"
        )


def write_rgba(classes_path: Path, colors: dict[int, str], rgba_path: Path):
    """Colorize the classified raster with `colors` as a tiled RGBA GeoTIFF in its own CRS.
    Values not in `colors`, and nodata, come out fully transparent."""
    with rasterio.open(classes_path) as src:
        values = src.read(1, masked=True)
        profile = src.profile

    rgba = np.zeros((4,) + values.shape, dtype=np.uint8)
    valid = ~np.ma.getmaskarray(values)
    for value, color in colors.items():
        rgba[:, valid & (values.data == value)] = np.array(hex_to_rgba(color), dtype=np.uint8)[:, None]

    profile.update(
        driver="GTiff",
        count=4,
        dtype="uint8",
        nodata=None,
        photometric="RGB",
        alpha="YES",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        bigtiff="IF_SAFER",
    )
    with rasterio.open(rgba_path, "w", **profile) as dst:
        dst.write(rgba)


def write_pmtiles(rgba_path: Path, mbtiles_path: Path, pmtiles_path: Path):
    """Pack an RGBA GeoTIFF into a PMTiles pyramid by way of MBTiles.

    The MBTiles driver warps the source to EPSG:3857 as part of the copy, and
    RESAMPLING=NEAREST governs that warp (keeping the flat category colors from blending at
    boundaries) as well as the overviews built after it. PNG8 (paletted) is plenty for a
    handful of flat colors and keeps tile size down.
    """
    rasterio.shutil.copy(rgba_path, mbtiles_path, driver="MBTILES", TILE_FORMAT="PNG8", RESAMPLING="NEAREST")
    with rasterio.open(mbtiles_path, "r+") as ds:
        ds.build_overviews(OVERVIEW_FACTORS, Resampling.nearest)
    pmtiles_path.unlink(missing_ok=True)
    mbtiles_to_pmtiles(str(mbtiles_path), str(pmtiles_path), None)


@click.command(help=__doc__)
@click.option("--year", type=int, required=True, help="calendar year to draw")
@click.option("--model-group-id", type=int, required=True, help="model group to draw")
@click.option(
    "--scenario-id",
    type=int,
    default=DEFAULT_SCENARIO_ID,
    show_default=True,
    help="scenario to draw",
)
@click.option(
    "--depths/--no-depths",
    default=False,
    show_default=True,
    help="color the water classes by depth, rather than drawing all water white",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="output directory for the PMTiles archives",
)
def main(year, model_group_id, scenario_id, depths, output):
    output.mkdir(parents=True, exist_ok=True)
    base = f"lnd_type_plus_mg{model_group_id}_s{scenario_id}_{year}"

    with tempfile.TemporaryDirectory(prefix="lnd_type_plus_") as tmp:
        work_dir = Path(tmp)
        classes_path = work_dir / "classes.tif"
        print(f"Running lnd_type_plus for model_group_id={model_group_id}, scenario_id={scenario_id}, {year}...")
        run_lnd_type_plus(model_group_id, scenario_id, year, classes_path)

        for group, colors in group_colors(depths).items():
            print(f"Building {group} tiles...")
            rgba_path = work_dir / f"{group}_rgba.tif"
            write_rgba(classes_path, colors, rgba_path)
            pmtiles_path = output / f"{base}_{group}.pmtiles"
            write_pmtiles(rgba_path, work_dir / f"{group}.mbtiles", pmtiles_path)
            print(f"Wrote {pmtiles_path}")


if __name__ == "__main__":
    main()
