from cpra.mp.data.raster import read_cog, vrt, write_cog  # noqa
from cpra.mp.data.tabular import compact_parquet, read_parquet, write_parquet
from cpra.mp.data.utils import read_fortran_array, validate_grid  # noqa


def compact_data(**kwargs):
    """Consolidate model data into larger files."""

    grid = validate_grid(**kwargs)

    if grid.grid_type == "vector":
        return compact_parquet(**kwargs)


def read_data(**kwargs):
    """Read model data from disk."""

    grid = validate_grid(**kwargs)

    if grid.grid_type == "raster":
        return read_cog(**kwargs)
    else:
        return read_parquet(**kwargs)


def write_data(data, **kwargs):
    """Write model data to disk."""

    grid = validate_grid(data, **kwargs)

    if grid.grid_type == "raster":
        return write_cog(data, **kwargs)
    else:
        return write_parquet(data, **kwargs)
