from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from cpra.mp.data.utils import read_fortran_array  # noqa


def write_data(data, **kwargs):
    """Write model data to disk."""

    if "geographic_unit" in kwargs and kwargs["geographic_unit"].endswith("_raster"):
        return write_cog(data, **kwargs)
    else:
        return write_parquet(data, **kwargs)
