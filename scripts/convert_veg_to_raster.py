import numpy as np
import polars as pl
import polars.selectors as cs
import rasterio as rio
import sys
from cpra.mp.data import read_data, write_data
from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.variables import VARIABLES


def convert_veg_to_raster(variable_name, grid_name, time_unit):
    """Convert vegetation grid data to raster images."""
    variable = VARIABLES[variable_name]
    numpy_dtype = pl.Series(dtype=variable.dtype).to_numpy().dtype

    with rio.open(
        BASE_PATH / "MP2023_S00_G000_C000_U00_V00_SLA_I_00_00_V_grid480.tif"
    ) as dataset:
        mask = dataset.dataset_mask().astype(numpy_dtype)
        mask[mask == 0] = -9999
        # The veg cell are numbered from bottom to top, left to right.
        flipped_mask = np.flip(mask, 0)

    lf = read_data(
        variable=variable_name, grid=grid_name, time_unit=time_unit
    ).with_columns(grid=pl.lit("veg_pixel_v001"))
    dims = lf.select(cs.matches(r"^[a-z]")).collect()

    for i, row in enumerate(lf.select(cs.digit()).collect().iter_rows()):
        flipped_data = flipped_mask.copy()
        flipped_data[flipped_data == 255] = row
        row_dims = dims[i].to_dicts()[0]

        print(f"Writing data for: {str(row_dims)}...")
        write_data(np.flip(flipped_data, 0), validate=False, **row_dims)


if __name__ == "__main__":
    convert_veg_to_raster(sys.argv[1], sys.argv[2], sys.argv[3])
