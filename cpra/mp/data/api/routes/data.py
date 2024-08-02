from cpra.mp.data import read_data
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from datetime import date
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException
from pydantic import create_model

data = APIRouter(tags=["Data and Analysis"])


class ResponseFormat(str, Enum):
    arrow = "arrow"
    csv = "csv"
    ndjson = "ndjson"
    parquet = "parquet"


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

dimension_params = {
    dim.name: (dimension_type(dim.name) | None, None)
    for dim in DIMENSIONS.values()
    if dim.type != "dataset"
}
DimensionsQuery = create_model("DimensionsQuery", **dimension_params)


@data.get("/{plan_version}/{variable}/{geographic_unit}/{time_unit}.{format}")
async def get_model_data(
    plan_version: PlanVersion,
    variable: Variable,
    geographic_unit: GeographicUnit,
    time_unit: TimeUnit,
    format: ResponseFormat,
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
    return PolarsStreamingResponse(lf, format)
