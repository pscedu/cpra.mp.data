from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from cpra.mp.data.utils import read_fortran_array, validate_grid  # noqa


def write_data(data, **kwargs):
    """Write model data to disk."""

    grid = validate_grid(data, **kwargs)

    if grid.grid_type == "raster":
        return write_cog(data, **kwargs)
    else:
        return write_parquet(data, **kwargs)
