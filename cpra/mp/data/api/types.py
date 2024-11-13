from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from datetime import date
from enum import Enum


def dimension_type(dim_name):
    """Create an enum with the values for a dimension."""
    dim = DIMENSIONS[dim_name]
    dtype_name = str(dim.dtype)
    if dim.values is not None:
        type_name = "".join(x.capitalize() for x in dim_name.lower().split("_"))
        return Enum(type_name, {str(v): v for v in dim.values["value"]})
    elif dtype_name.startswith("Int"):
        return int
    elif dtype_name.startswith("Float"):
        return float
    elif dtype_name.startswith("Date"):
        return date
    return str


class AggregateFunction(str, Enum):
    max = "max"
    mean = "mean"
    min = "min"
    sum = "sum"


class CompressionAlgorithm(str, Enum):
    deflate = "deflate"
    gzip = "gzip"
    lz4 = "lz4"
    lzma = "lzma"
    lzw = "lzw"
    none = "none"
    zstd = "zstd"


class DataFormat(str, Enum):
    # arrow = "arrow"
    csv = "csv"
    ndjson = "ndjson"
    parquet = "parquet"
    tif = "tif"


class OutputType(str, Enum):
    raster = "raster"
    vector = "vector"


class TileFormat(str, Enum):
    jpg = "jpg"
    png = "png"


Variable = dimension_type("variable")
Grid = dimension_type("grid")
TimeUnit = dimension_type("time_unit")
ColorMap = Enum("ColorMap", {str(k): k for k in LEGENDS.keys()})
VectorGrid = Enum(
    "VectorGrid",
    {str(k): k for k, v in GRIDS.items() if v.grid_type == "vector"},
)
