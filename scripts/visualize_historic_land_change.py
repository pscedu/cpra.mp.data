from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from cpra.mp.data import read_data
from cpra.mp.data.grids import GRIDS

# Two-digit code: (2019 lnd_type) * 10 + (historic lnd_type)
# 1 = land, 2 = water; e.g. 21 = land in 2019 / water historically (gain),
# 12 = water in 2019 / land historically (loss).
COLOR_MAP = {
    55: (0, 0, 0),
    52: (69, 145, 172),
    51: (0, 0, 0),
    44: (0, 0, 0),
    33: (0, 0, 0),
    32: (69, 145, 172),
    31: (0, 0, 0),
    23: (38, 81, 24),
    22: (0, 0, 0),
    21: (38, 81, 24),
    13: (0, 0, 0),
    12: (69, 145, 172),
    11: (0, 0, 0),
    0: (0, 0, 0),
}

YEARS = [
    1932, 1956, 1973, 1975, 1977, 1985, 1988, 1990,
    1995, 1998, 1999, 2002, 2004, 2006, 2008, 2009,
    2010, 2013, 2014, 2015,
]


def get_raster_path(**kwargs):
    return read_data(**kwargs).collect().item(0, "path")


@click.command()
@click.argument("output_dir", type=click.Path(file_okay=False))
def main(output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = GRIDS["morph_pixel_v001"]
    dst_crs = CRS.from_user_input(grid.crs)
    dst_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)

    dst_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 4,
        "crs": dst_crs,
        "transform": dst_transform,
        "width": grid.width,
        "height": grid.height,
        "compress": "zstd",
    }

    ref_path = get_raster_path(
        variable="lnd_type",
        grid="morph_pixel_v001",
        time_unit="annual",
        model_group_id=500,
        scenario_id=7,
        calendar_year=2019,
    )
    with rasterio.open(ref_path) as src:
        ref = src.read(1)

    for year in YEARS:
        print(f"Processing year {year}...")

        hist_path = get_raster_path(
            variable="lnd_type",
            grid="morph_pixel_v001",
            time_unit="annual",
            model_group_id=0,
            scenario_id=0,
            calendar_year=year,
        )
        with rasterio.open(hist_path) as src:
            hist = src.read(1)

        # Combine 2019 and historic values into a two-digit code.
        # Nodata pixels (-9999) in either layer fall through to code 0 → black.
        valid = (ref > 0) & (hist > 0)
        code = np.where(valid, ref * 10 + hist, 0).astype(np.int32)

        rgba = np.zeros((4, grid.height, grid.width), dtype=np.uint8)
        for code_val, (r, g, b) in COLOR_MAP.items():
            if (r, g, b) == (0, 0, 0):
                continue
            mask = code == code_val
            rgba[0, mask] = r
            rgba[1, mask] = g
            rgba[2, mask] = b
            rgba[3, mask] = 255

        out_path = out_dir / f"historical_land_change_{year}.tif"
        with rasterio.open(out_path, "w", **dst_profile) as dst:
            dst.write(rgba)


if __name__ == "__main__":
    main()
