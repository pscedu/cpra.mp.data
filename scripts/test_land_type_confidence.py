import asyncio
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from cpra.mp.data import read_cog
from cpra.mp.data.api.utils import parallel_read_rasters
from cpra.mp.data.grids import GRIDS

# Maps lnd_type values to confidence output values.
# 1/3/4 = land-like → 1 (confident land)
# 2/5 = water-like → 0 (confident water)
# anything else (including nodata) → -9999
VALUE_MAP = {
    1: 1,
    2: 0,
    3: 1,
    4: 1,
    5: 0,
}

NODATA = -9999


def apply_value_map(arr):
    out = np.full(arr.shape, NODATA, dtype=np.float32)
    for in_val, out_val in VALUE_MAP.items():
        out[arr == in_val] = out_val
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

    # Historic data (model_group_id=0, scenario_id=0) — output as int32
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

    # Current data (model_group_id=500) — combine scenario 7 (60%) and 8 (40%), output as float32
    def load_scenario(scenario_id):
        return (
            read_cog(
                variable="lnd_type",
                grid="morph_pixel_v001",
                time_unit="annual",
                model_group_id=500,
                scenario_id=scenario_id,
            )
            .collect()
            .select(["calendar_year", "path"])
        )

    current_inputs = load_scenario(7).join(load_scenario(8), on="calendar_year", suffix="_s8").sort("calendar_year")

    for row in current_inputs.iter_rows(named=True):
        year = row["calendar_year"]
        out_path = out_dir / f"test_land_type_confidence_{year}.tif"
        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
            continue
        print(f"Processing current year {year}...")

        with rasterio.open(row["path"]) as s7_src, rasterio.open(row["path_s8"]) as s8_src:
            s7, s8 = [r[0] for r in asyncio.run(parallel_read_rasters([s7_src, s8_src]))]

        mapped_s7 = apply_value_map(s7)
        mapped_s8 = apply_value_map(s8)

        valid = (mapped_s7 != NODATA) & (mapped_s8 != NODATA)
        out = np.where(valid, 0.6 * mapped_s7 + 0.4 * mapped_s8, NODATA).astype(np.float32)

        with rasterio.open(out_path, "w", **base_profile) as dst:
            dst.write(out, 1)


if __name__ == "__main__":
    main()
