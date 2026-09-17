"""Enums the request layer builds a `Pipeline` out of: dataset dimensions turned into
types (`Variable`/`Grid`/`TimeUnit`), the aggregate vocabulary (`AggregateFunction`), and
the output-format vocabulary (`DataFormat`/`TileFormat`/`CompressionAlgorithm`/
`OutputType`). `request/aggregates.py` is the sibling module that owns what each
aggregate NAME means legally (which step it may appear on, whether it folds); this module
only owns the closed set of names and format identifiers themselves.
"""

from datetime import date
from enum import Enum

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS, WGS84


def dimension_type(dim_name):
    """Build the Python type a dimension's declared values are checked against: `int`/
    `float`/`date` for a numeric/date dtype, or a dynamically-built string `Enum` (one
    member per declared value) for anything else with a known value set, falling back to
    a bare `str` when neither applies.
    """
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
    """The closed aggregate vocabulary a `project`/`group` step may name. `count`, `max`,
    `mean`, `min`, and `sum` collapse a group to one row and are what `group.aggregate`
    accepts; `count_by_value` collapses instead to one row per (group, distinct value) --
    a different output shape from the other five, legal only where that shape has
    somewhere to go (`project`, whose target grid's cells absorb the extra rows -- see
    `request/aggregates.py` for exactly which steps accept which aggregate, and why
    `count_by_value` is excluded from `group`).
    """

    count = "count"
    max = "max"
    mean = "mean"
    min = "min"
    sum = "sum"
    count_by_value = "count_by_value"


class CompressionAlgorithm(str, Enum):
    deflate = "deflate"
    gzip = "gzip"
    lz4 = "lz4"
    lzma = "lzma"
    lzw = "lzw"
    none = "none"
    zstd = "zstd"


class DataFormat(str, Enum):
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
# only ever a source -- a coordinate is already what it converts to.
CrosswalkGrid = Enum(
    "CrosswalkGrid",
    {WGS84.name: WGS84.name, **{str(k): k for k in GRIDS}},
)
