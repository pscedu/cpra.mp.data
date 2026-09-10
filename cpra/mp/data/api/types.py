from datetime import date
from enum import Enum

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS, WGS84
from cpra.mp.data.legends import LEGENDS


def dimension_type(dim_name):
    """Create an enum with the values for a dimension."""
    dim = DIMENSIONS[dim_name]
    dtype_name = str(dim.dtype)
    if dtype_name.startswith("Int"):
        return int
    elif dtype_name.startswith("Float"):
        return float
    elif dtype_name.startswith("Date"):
        return date
    elif dim.values is not None:
        # Only create enums for string dimensions for now. It looks like
        # it's possible to get Pydantic/FastAPI to work with non-string
        # enums, but it requires some gymnastics. See:
        # https://github.com/fastapi/fastapi/issues/3247
        type_name = "".join(x.capitalize() for x in dim_name.lower().split("_"))
        return Enum(type_name, {str(v): v for v in dim.values["value"]})
    return str


class AggregateFunction(str, Enum):
    # `count` is valid as a `summarize_aggregate`/`group_by`-vertical-reduce value on
    # both grid types (see raster_source.POLARS_RASTER_AGGREGATES and the vector
    # `getattr(pl.col("value"), aggregate)()` call sites, both of which resolve it to
    # Polars' own null-skipping `Expr.count`), but NOT as a `group_aggregate` (the raster
    # band-COLLAPSE step, `raster_source.HORIZONTAL_RASTER_AGGREGATES`): "how many of
    # these bands contributed to this pixel" has no natural per-pixel scalar meaning the
    # way max/mean/min/sum do, and `Dataset` rejects it there with an explicit error
    # rather than let it fail deeper as a raw `KeyError`/`AttributeError`.
    count = "count"
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
VectorGrid = Enum(
    "VectorGrid",
    {str(k): k for k, v in GRIDS.items() if v.grid_type == "vector"},
)

# The grids a crosswalk can convert from: every grid, plus the WGS84 pseudo-grid, which is
# only ever a source — a coordinate is already what it converts to.
CrosswalkGrid = Enum(
    "CrosswalkGrid",
    {WGS84.name: WGS84.name, **{str(k): k for k in GRIDS}},
)
