from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import read_parquet, write_parquet
from cpra.mp.data.utils import read_fortran_array, validate_grid  # noqa


def read_data(**kwargs):
    """Read model data from disk."""

    grid = validate_grid(**kwargs)

    if grid.grid_type == "raster":
        msg = "Reading raster data is not yet implemented"
        raise NotImplementedError(msg)
    else:
        return read_parquet(**kwargs)


def write_data(data, **kwargs):
    """Write model data to disk."""

    grid = validate_grid(data, **kwargs)

    if grid.grid_type == "raster":
        return write_cog(data, **kwargs)
    else:
        return write_parquet(data, **kwargs)
