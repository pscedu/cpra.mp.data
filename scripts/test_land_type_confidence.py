import asyncio
from pathlib import Path

import click
import numpy as np
import polars as pl
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import Window

from cpra.mp.data import read_cog
from cpra.mp.data.api.utils import parallel_read_rasters
from cpra.mp.data.grids import GRIDS

# Maps lnd_type values to confidence output values.
# 1/3/4 = land-like → 1 (confident land)
# 2/5 = water-like → 0 (confident water)
# anything else (including nodata) → NaN / -9999
VALUE_MAP = {
    1: 1,
    2: 0,
    3: 1,
    4: 1,
    5: 0,
}

NODATA = -9999
TILE_SIZE = 512


def apply_value_map(arr):
    out = np.full(arr.shape, NODATA, dtype=np.float32)
    for in_val, out_val in VALUE_MAP.items():
        out[arr == in_val] = out_val
    return out


def to_binary_nan(arr):
    """Map lnd_type to 0.0/1.0, using NaN for unmapped/nodata values."""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for in_val, out_val in VALUE_MAP.items():
        out[arr == in_val] = float(out_val)
    return out


@click.command()
@click.argument("output_dir", type=click.Path(file_okay=False))
def main(output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = GRIDS["morph_pixel_v001"]
    dst_crs = CRS.from_user_input(grid.crs)
    dst_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)

    base_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "width": grid.width,
        "height": grid.height,
        "compress": "zstd",
        "nodata": NODATA,
    }

    # Historic data (model_group_id=0, scenario_id=0)
    hist_inputs = (
        read_cog(
            variable="lnd_type",
            grid="morph_pixel_v001",
            time_unit="annual",
            model_group_id=0,
            scenario_id=0,
        )
        .collect()
        .sort("calendar_year")
    )

    for row in hist_inputs.iter_rows(named=True):
        year = row["calendar_year"]
        out_path = out_dir / f"test_land_type_confidence_{year}.tif"
        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
            continue
        print(f"Processing historic year {year}...")

        with rasterio.open(row["path"]) as src:
            hist = src.read(1)

        out = apply_value_map(hist)

        with rasterio.open(out_path, "w", **base_profile) as dst:
            dst.write(out, 1)

    # Current data — all model groups (excluding model_group_id=0), all scenarios.
    # Convert each raster to 0/1 and average across all members for each year.
    year_paths = (
        read_cog(
            variable="lnd_type",
            grid="morph_pixel_v001",
            time_unit="annual",
        )
        .collect()
        .filter(pl.col("model_group_id") != 0)
        .group_by("calendar_year")
        .agg(pl.col("path").sort())
        .sort("calendar_year")
    )

    for row in year_paths.iter_rows(named=True):
        year = row["calendar_year"]
        paths = row["path"]
        out_path = out_dir / f"test_land_type_confidence_{year}.tif"
        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
            continue
        print(f"Processing year {year} ({len(paths)} rasters)...")

        srcs = [rasterio.open(p) for p in paths]
        try:
            with rasterio.open(out_path, "w", **base_profile) as dst:
                for row_off in range(0, grid.height, TILE_SIZE):
                    tile_h = min(TILE_SIZE, grid.height - row_off)
                    for col_off in range(0, grid.width, TILE_SIZE):
                        tile_w = min(TILE_SIZE, grid.width - col_off)
                        window = Window(col_off, row_off, tile_w, tile_h)

                        tiles = asyncio.run(parallel_read_rasters(srcs, window=window))
                        mapped = np.stack([to_binary_nan(t[0]) for t in tiles], axis=0)

                        valid = ~np.isnan(mapped)
                        count = valid.sum(axis=0)
                        total = np.where(valid, mapped, 0.0).sum(axis=0)
                        mean = np.zeros(count.shape, dtype=np.float32)
                        np.divide(total, count, out=mean, where=count > 0)
                        out = np.where(count > 0, mean, NODATA).astype(np.float32)

                        dst.write(out, 1, window=window)
        finally:
            for src in srcs:
                src.close()


if __name__ == "__main__":
    main()
