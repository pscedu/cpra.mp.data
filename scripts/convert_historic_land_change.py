import click
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from cpra.mp.data import read_data, write_data
from cpra.mp.data.grids import GRIDS


@click.command()
@click.argument("historic_image", type=click.Path(exists=True))
def main(historic_image):
    loss_breakpoints = list(range(20))
    years = [
        1932,
        1956,
        1973,
        1975,
        1977,
        1985,
        1988,
        1990,
        1995,
        1998,
        1999,
        2002,
        2004,
        2006,
        2008,
        2009,
        2010,
        2013,
        2014,
        2015,
    ]
    gain_start = 20

    # The historic image is at 15m resolution. Resample it to the morph_pixel_v001
    # grid (30m, EPSG:26915) so its pixels align 1:1 with the 2019 reference raster.
    # Nearest neighbor preserves the original integer class values without interpolation.
    grid = GRIDS["morph_pixel_v001"]
    dst_crs = CRS.from_user_input(grid.crs)
    dst_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)

    historic = np.empty((grid.height, grid.width), dtype=np.uint8)
    with rasterio.open(historic_image) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=historic,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    ref_path = (
        read_data(
            variable="lnd_type",
            grid="morph_pixel_v001",
            time_unit="annual",
            model_group_id=500,
            scenario_id=7,
            calendar_year=2019,
        )
        .collect()
        .item(0, "path")
    )

    with rasterio.open(ref_path) as src:
        ref = src.read(1)

    for loss_breakpoint, year in zip(loss_breakpoints, years):
        gain_breakpoint = gain_start + loss_breakpoint - 1

        print(f"Converting year {year}...")

        # Land not yet lost: historic values in (loss_breakpoint, 19]
        green_mask = (historic > loss_breakpoint) & (historic <= 19)
        # Land gain already occurred: historic values in (gain_breakpoint, 9998]
        blue_mask = (historic > gain_breakpoint) & (historic <= 9998)

        output = ref.copy()
        output[green_mask & (ref == 2)] = 1
        output[blue_mask] = 2

        write_data(
            output,
            variable="lnd_type",
            grid="morph_pixel_v001",
            time_unit="annual",
            model_group_id=0,
            scenario_id=0,
            calendar_year=year,
            model="icm_v2023.0.0",
        )


if __name__ == "__main__":
    main()
