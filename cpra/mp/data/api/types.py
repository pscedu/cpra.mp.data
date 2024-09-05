from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from datetime import date
from enum import Enum
from pydantic import create_model, ConfigDict


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
    difference = "difference"
    max = "max"
    mean = "mean"
    min = "min"
    sum = "sum"


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


PlanVersion = dimension_type("plan_version")
Variable = dimension_type("variable")
GeographicUnit = dimension_type("geographic_unit")
TimeUnit = dimension_type("time_unit")
ColorMap = Enum("ColorMap", {str(k): k for k in LEGENDS.keys()})
VectorGeographicUnit = Enum(
    "GeographicUnit",
    {str(k): k for k, v in GRIDS.items() if list(v.values())[0].grid_type == "vector"},
)

dimension_params = {
    dim.name: (dimension_type(dim.name) | None, None)
    for dim in DIMENSIONS.values()
    if dim.type != "dataset"
}
DimensionsQuery = create_model(
    "DimensionsQuery",
    model_config=ConfigDict(protected_namespaces=()),
    **dimension_params,
)
