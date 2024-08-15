from cpra.mp.data import read_data
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.api.raster import RasterioStreamingResponse
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from datetime import date
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException
from pydantic import create_model, ConfigDict

data = APIRouter(tags=["Data and Analysis"])


class AggregateFunction(str, Enum):
    difference = "difference"
    max = "max"
    mean = "mean"
    min = "min"
    sum = "sum"


class ResponseFormat(str, Enum):
    # arrow = "arrow"
    csv = "csv"
    ndjson = "ndjson"
    parquet = "parquet"
    tif = "tif"


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


@data.get("/{plan_version}/{variable}/{geographic_unit}/{time_unit}.{format}")
async def get_model_data(
    plan_version: PlanVersion,
    variable: Variable,
    geographic_unit: GeographicUnit,
    time_unit: TimeUnit,
    format: ResponseFormat,
    geography: str | None = None,
    where: str | None = None,
    presort: str | None = None,
    group_by: str | None = None,
    group_aggregate: AggregateFunction | None = None,
    color_map: ColorMap | None = None,
    melt: bool = False,
    summarize_by: VectorGeographicUnit | None = None,
    summarize_aggregate: AggregateFunction | None = None,
    summarize_geography: str | None = None,
    sort: str | None = None,
    dimensions: DimensionsQuery = Depends(),
):
    dimension_kwargs = {
        k: (v.value if isinstance(v, Enum) else v)
        for k, v in dimensions
        if v is not None
    }
    try:
        lf = read_data(
            plan_version=plan_version.value,
            variable=variable.value,
            geographic_unit=geographic_unit.value,
            time_unit=time_unit.value,
            **dimension_kwargs,
        )
    except ValidationError as err:
        # TODO: Also handle directory does not exist errors.
        raise HTTPException(status_code=422, detail=str(err))

    # The presence of a path column indicates that this is a raster.
    if "path" in lf.columns:
        return RasterioStreamingResponse(lf, format)
    return PolarsStreamingResponse(lf, format)
